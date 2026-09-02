add_rules("mode.debug", "mode.release")
add_rules("plugin.compile_commands.autoupdate", {outputdir = "."})
set_languages("c++latest")

set_policy("build.sanitizer.address", true)
add_requires("eigen 5.0.1","taocpp-json 2025.03.11","kompute v0.9.0")
--add_requires("boost",{config = {cmake = false}})
target("simu")
    set_kind("binary")
    add_files("src/*.cpp")
    add_files("src/*.cxx")
    add_packages("eigen","taocpp-json","kompute")
target_end()



package("kompute")

    set_homepage("https://github.com/KomputeProject/kompute")
    set_description("General purpose GPU compute framework for cross vendor graphics cards")
    set_license("Apache-2.0")

    add_urls("https://github.com/KomputeProject/kompute.git")
	add_versions("v0.9.0","f3c7bd02fbe5fcd27b3810ec585176c59083d373")

    add_deps("cmake", "vulkan-loader")

    on_install("windows|x86", "windows|x64", "linux", "macosx|!arm64", function (package)
        local configs = {}
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
        -- v0.9.0 removed KOMPUTE_OPT_REPO_SUBMODULE_BUILD and KOMPUTE_OPT_BUILD_AS_SHARED_LIB
        -- (both now trigger a fatal error in cmake/deprecation_warnings.cmake).
        -- Shared/static is controlled via BUILD_SHARED_LIBS, which xmake's cmake tool
        -- passes automatically from package:config("shared").
        table.insert(configs, "-DKOMPUTE_OPT_INSTALL=ON")
        table.insert(configs, "-DKOMPUTE_OPT_BUILD_TESTS=OFF")
        table.insert(configs, "-DKOMPUTE_OPT_BUILD_DOCS=OFF")
        -- don't require a GPU/vulkaninfo at package build time
        table.insert(configs, "-DKOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK=ON")
        import("package.tools.cmake").install(package, configs)
        -- headers, fmt and cmake config files are installed by KOMPUTE_OPT_INSTALL;
        -- the old manual os.cp("single_include"/"external/fmt/include") no longer exist in v0.9.0
    end)

    on_test(function (package)
        assert(package:has_cxxtypes("kp::Manager", {includes = "kompute/Kompute.hpp"}))
    end)
package_end()

--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro definition
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--
