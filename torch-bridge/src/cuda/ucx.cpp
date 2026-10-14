#include <ucp/api/ucp.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
thread_local std::string error;
void check(ucs_status_t status) {
  if(status!=UCS_OK) throw std::runtime_error(ucs_status_string(status));
}
template<class F> int guarded(F&& f) {
  try {f();return 0;} catch(const std::exception& e) {error=e.what();return -1;}
  catch(...) {error="unknown UCX transport failure";return -1;}
}
struct Worker {
  ucp_context_h context=nullptr;
  ucp_worker_h worker=nullptr;
  std::vector<ucp_ep_h> peers;
  std::atomic<int> failure{UCS_OK};
  ~Worker() {
    if(worker) {
      ucp_request_param_t param{};
      param.op_attr_mask=UCP_OP_ATTR_FIELD_FLAGS;
      param.flags=UCP_EP_CLOSE_FLAG_FORCE;
      for(auto ep:peers) if(ep) ucp_ep_close_nbx(ep,&param);
      ucp_worker_destroy(worker);
    }
    if(context) ucp_cleanup(context);
  }
};
void endpoint_error(void* user,ucp_ep_h,ucs_status_t status) {
  static_cast<Worker*>(user)->failure.store(status,std::memory_order_release);
}
struct Pending {
  std::atomic<int> status{UCS_INPROGRESS};
  size_t expected=0;
  void* request=nullptr;
};
void sent(void*,ucs_status_t status,void* user) {
  static_cast<Pending*>(user)->status.store(status,std::memory_order_release);
}
void received(void*,ucs_status_t status,const ucp_tag_recv_info_t* info,void* user) {
  auto& pending=*static_cast<Pending*>(user);
  if(status==UCS_OK && (!info || info->length!=pending.expected)) status=UCS_ERR_MESSAGE_TRUNCATED;
  pending.status.store(status,std::memory_order_release);
}
}

extern "C" {
struct RucclUcxTransfer {uint32_t peer,receive;uint64_t tag,pointer;size_t bytes;};
const char* ruccl_ucx_c_error() {return error.c_str();}

int ruccl_ucx_c_create(uint32_t world,const char* transports,void** output) {
  return guarded([&] {
    if(!world || !output || !transports || !*transports) throw std::runtime_error("invalid UCX creation arguments");
    auto w=std::make_unique<Worker>();
    ucp_config_t* raw=nullptr;check(ucp_config_read(nullptr,nullptr,&raw));
    std::unique_ptr<ucp_config_t,decltype(&ucp_config_release)> config(raw,ucp_config_release);
    check(ucp_config_modify(raw,"TLS",transports));
    // Payloads use rendezvous zero-copy; control setup uses the PyTorch Store.
    check(ucp_config_modify(raw,"RNDV_THRESH","0"));
    check(ucp_config_modify(raw,"RNDV_SCHEME","put_zcopy"));
    ucp_params_t params{};
    params.field_mask=UCP_PARAM_FIELD_FEATURES;
    params.features=UCP_FEATURE_TAG;
    check(ucp_init(&params,raw,&w->context));
    ucp_context_attr_t attributes{};attributes.field_mask=UCP_ATTR_FIELD_MEMORY_TYPES;
    check(ucp_context_query(w->context,&attributes));
    if(!(attributes.memory_types&(uint64_t(1)<<UCS_MEMORY_TYPE_CUDA)))
      throw std::runtime_error("UCX was built without CUDA memory support");
    ucp_worker_params_t worker_params{};
    worker_params.field_mask=UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode=UCS_THREAD_MODE_MULTI;
    check(ucp_worker_create(w->context,&worker_params,&w->worker));
    ucp_worker_attr_t worker_attr{};worker_attr.field_mask=UCP_WORKER_ATTR_FIELD_THREAD_MODE;
    check(ucp_worker_query(w->worker,&worker_attr));
    if(worker_attr.thread_mode!=UCS_THREAD_MODE_MULTI)
      throw std::runtime_error("UCX multi-thread worker support is required for concurrent pipeline traffic");
    w->peers.resize(world,nullptr);
    *output=w.release();
  });
}

int ruccl_ucx_c_address(void* handle,void* destination,size_t* length) {
  return guarded([&] {
    auto& w=*static_cast<Worker*>(handle);
    ucp_address_t* address=nullptr;size_t count=0;
    check(ucp_worker_get_address(w.worker,&address,&count));
    if(destination && *length<count) {
      ucp_worker_release_address(w.worker,address);
      throw std::runtime_error("UCX worker address buffer too small");
    }
    if(destination) std::memcpy(destination,address,count);
    *length=count;ucp_worker_release_address(w.worker,address);
  });
}

int ruccl_ucx_c_connect(void* handle,uint32_t peer,const void* address) {
  return guarded([&] {
    auto& w=*static_cast<Worker*>(handle);
    if(peer>=w.peers.size() || w.peers[peer] || !address) throw std::runtime_error("invalid UCX peer");
    ucp_ep_params_t params{};
    params.field_mask=UCP_EP_PARAM_FIELD_REMOTE_ADDRESS|UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE|UCP_EP_PARAM_FIELD_ERR_HANDLER;
    params.address=static_cast<const ucp_address_t*>(address);
    params.err_mode=UCP_ERR_HANDLING_MODE_PEER;
    params.err_handler.cb=endpoint_error;params.err_handler.arg=&w;
    check(ucp_ep_create(w.worker,&params,&w.peers[peer]));
  });
}

int ruccl_ucx_c_exchange(void* handle,const RucclUcxTransfer* transfers,size_t count,uint64_t timeout_ms) {
  return guarded([&] {
    auto& w=*static_cast<Worker*>(handle);
    if((count && !transfers) || !timeout_ms) throw std::runtime_error("invalid UCX transfer list/timeout");
    check(static_cast<ucs_status_t>(w.failure.load(std::memory_order_acquire)));
    for(size_t i=0;i<count;++i) {
      const auto& t=transfers[i];
      if(t.peer>=w.peers.size() || !w.peers[t.peer] || t.receive>1 || (!t.pointer && t.bytes))
        throw std::runtime_error("invalid UCX transfer descriptor");
    }
    auto pending=std::make_unique<Pending[]>(count);
    ucs_status_t failure=UCS_OK;
    for(size_t i=0;i<count;++i) {
      const auto& t=transfers[i];auto& p=pending[i];p.expected=t.bytes;
      ucp_request_param_t param{};
      param.op_attr_mask=UCP_OP_ATTR_FIELD_CALLBACK|UCP_OP_ATTR_FIELD_USER_DATA|UCP_OP_ATTR_FIELD_MEMORY_TYPE
                        |UCP_OP_ATTR_FIELD_DATATYPE|UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
      param.user_data=&p;param.memory_type=UCS_MEMORY_TYPE_CUDA;param.datatype=ucp_dt_make_contig(1);
      if(t.receive) {
        param.cb.recv=received;
        p.request=ucp_tag_recv_nbx(w.worker,reinterpret_cast<void*>(uintptr_t(t.pointer)),t.bytes,t.tag,UINT64_MAX,&param);
      } else {
        param.cb.send=sent;
        p.request=ucp_tag_send_nbx(w.peers[t.peer],reinterpret_cast<const void*>(uintptr_t(t.pointer)),t.bytes,t.tag,&param);
      }
      if(UCS_PTR_IS_ERR(p.request)) {
        failure=UCS_PTR_STATUS(p.request);p.request=nullptr;p.status.store(failure);
      } else if(!p.request) p.status.store(UCS_OK);
    }
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
    bool canceled=false;
    for(;;) {
      bool complete=true;
      for(size_t i=0;i<count;++i) {
        const int status=pending[i].status.load(std::memory_order_acquire);
        if(status==UCS_INPROGRESS) complete=false;
        else if(status!=UCS_OK && failure==UCS_OK) failure=static_cast<ucs_status_t>(status);
      }
      if(complete) break;
      const int ep_failure=w.failure.load(std::memory_order_acquire);
      if(failure==UCS_OK && ep_failure!=UCS_OK) failure=static_cast<ucs_status_t>(ep_failure);
      if(failure==UCS_OK && std::chrono::steady_clock::now()>=deadline) failure=UCS_ERR_TIMED_OUT;
      if(failure!=UCS_OK && !canceled) {
        canceled=true;
        for(size_t i=0;i<count;++i) if(pending[i].request && pending[i].status.load()==UCS_INPROGRESS)
          ucp_request_cancel(w.worker,pending[i].request);
      }
      if(!ucp_worker_progress(w.worker)) std::this_thread::yield();
    }
    // Drain cancellation before releasing request state or caller GPU buffers.
    for(size_t i=0;i<count;++i) if(pending[i].request) ucp_request_free(pending[i].request);
    check(failure);
  });
}
void ruccl_ucx_c_destroy(void* handle) {delete static_cast<Worker*>(handle);}
}
