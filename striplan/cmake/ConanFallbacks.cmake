function(conan_add_package_include_fallback target package_prefix)
    set(_fallback_includes "")
    if(DEFINED ${package_prefix}_PACKAGE_FOLDER_DEBUG)
        list(APPEND _fallback_includes
            "$<$<CONFIG:Debug>:${${package_prefix}_PACKAGE_FOLDER_DEBUG}/include>")
    endif()
    if(DEFINED ${package_prefix}_PACKAGE_FOLDER_RELEASE)
        list(APPEND _fallback_includes
            "$<$<CONFIG:Release>:${${package_prefix}_PACKAGE_FOLDER_RELEASE}/include>")
    endif()
    if(DEFINED ${package_prefix}_PACKAGE_FOLDER_RELWITHDEBINFO)
        list(APPEND _fallback_includes
            "$<$<CONFIG:RelWithDebInfo>:${${package_prefix}_PACKAGE_FOLDER_RELWITHDEBINFO}/include>")
    endif()
    if(DEFINED ${package_prefix}_PACKAGE_FOLDER_MINSIZEREL)
        list(APPEND _fallback_includes
            "$<$<CONFIG:MinSizeRel>:${${package_prefix}_PACKAGE_FOLDER_MINSIZEREL}/include>")
    endif()
    if(DEFINED ${package_prefix}_PACKAGE_FOLDER)
        list(APPEND _fallback_includes "${${package_prefix}_PACKAGE_FOLDER}/include")
    endif()

    if(_fallback_includes)
        target_include_directories(${target} SYSTEM PUBLIC ${_fallback_includes})
    endif()
endfunction()
