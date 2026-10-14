fn main() {
    #[cfg(feature="ucx")]
    {
        assert!(std::env::var("CARGO_CFG_TARGET_OS").unwrap()=="linux","ruCCL UCX requires Linux");
        let library=pkg_config::Config::new().atleast_version("1.15").probe("ucx")
            .expect("ruCCL UCX needs UCX >= 1.15 development files (pkg-config ucx)");
        let mut build=cc::Build::new();
        build.cpp(true).std("c++17").file("src/cuda/ucx.cpp");
        for path in library.include_paths {build.include(path);}
        build.compile("ruccl_ucx_transport");
        println!("cargo:rerun-if-changed=src/cuda/ucx.cpp");
    }
}
