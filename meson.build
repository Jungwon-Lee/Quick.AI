project('quick_dot_ai', 'c', 'cpp',
  version: '0.1.0',
  license: ['apache-2.0'],
  meson_version: '>=0.55.0',
  default_options: [
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++17',
    'buildtype=release',
  ],
)

# Pull in nntrainer via the bundled submodule as a meson subproject.
# We disable Applications so that nntrainer's own (now legacy) in-tree
# Applications/CausalLM is not built alongside this standalone one, and
# we skip tests to keep the dependency build lean.
nntrainer_proj = subproject('nntrainer',
  default_options: [
    'enable-app=false',
    'enable-test=false',
    'enable-tflite-backbone=false',
    'enable-tflite-interpreter=false',
    'werror=false',
  ],
)

nntrainer_dep       = nntrainer_proj.get_variable('nntrainer_dep')
nntrainer_ccapi_dep = nntrainer_proj.get_variable('nntrainer_ccapi_dep')
application_install_dir = get_option('prefix') / get_option('libdir') / 'quick_dot_ai'

openmp_dep = dependency('openmp')
powershell_prog = find_program('powershell', required: (build_machine.system() == 'windows'))

quick_dot_ai_src = [
    meson.current_source_dir() / 'huggingface_tokenizer.cpp',
    meson.current_source_dir() / 'llm_util.cpp',
    meson.current_source_dir() / 'api' / 'causal_lm_api.cpp',
    meson.current_source_dir() / 'api' / 'model_config.cpp',
]

executable_src = [
    meson.current_source_dir() / 'main.cpp',
    meson.current_source_dir() / 'chat_template.cpp',
]

quick_dot_ai_inc_abs = [meson.current_source_dir()]
quick_dot_ai_inc = [include_directories('.')]
quick_dot_ai_minja_inc = include_directories('third_party/minja/include')

# Build layer dependency
quick_dot_ai_layer_dependencies = []
subdir('layers')

# Add common layers to dependencies
quick_dot_ai_layer_dependencies += [
    quick_dot_ai_rms_norm_dep,
    quick_dot_ai_tie_word_embedding_dep,
    quick_dot_ai_lm_head_dep,
    quick_dot_ai_swiglu_dep,
    quick_dot_ai_mha_core_dep,
    quick_dot_ai_embedding_layer_dep,
    quick_dot_ai_lm_head_dep,
    quick_dot_ai_reshaped_rms_norm_dep,
    quick_dot_ai_qkv_layer_dep,
    quick_dot_ai_embedding_pooling_layer_dep,
    quick_dot_ai_embedding_normalize_layer_dep,
]

subdir('models')

if (get_option('platform') == 'windows') and (build_machine.system() == 'windows')
    run_command(powershell_prog, '-ExecutionPolicy', 'Bypass', '-File', join_paths(meson.current_source_dir(), 'jni', 'prepare_encoder.ps1'), meson.build_root(), '0.2', check: true)
elif get_option('platform') != 'tizen'
    run_command([meson.current_source_dir() / 'jni' / 'prepare_encoder.sh', meson.build_root(), '0.2'], check: true)
endif

tokenizer_dependencies = []
tokenizer_path = meson.current_source_dir() / 'lib' / 'libtokenizers_c.a'

cpp_args = []

if get_option('default_library') == 'shared'
    cpp_args += [ '-DPLUGGABLE' ]
endif

quick_dot_ai = shared_library('quick_dot_ai',
    quick_dot_ai_src,
    dependencies: [
        nntrainer_dep,
        nntrainer_ccapi_dep,
        quick_dot_ai_layer_dependencies,
        tokenizer_dependencies,
        openmp_dep
    ],
    include_directories: quick_dot_ai_inc,
    install: true,
    install_dir: application_install_dir,
    link_args: [tokenizer_path],
    cpp_args: cpp_args
)

quick_dot_ai_dep = declare_dependency(
    link_with: [quick_dot_ai],
    include_directories: quick_dot_ai_inc
)

e = executable('quick_dot_ai_run',
    executable_src,
    include_directories: [quick_dot_ai_inc, quick_dot_ai_minja_inc],
    dependencies: [
        nntrainer_dep,
        nntrainer_ccapi_dep,
        quick_dot_ai_layer_dependencies,
        quick_dot_ai_dep
    ],
)


quantize_src = [
    meson.current_source_dir() / 'quantize.cpp',
]

e_quantize = executable('quick_dot_ai_quantize',
    quantize_src,
    include_directories: quick_dot_ai_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, quick_dot_ai_layer_dependencies, quick_dot_ai_dep],
)


test_api_src = [
    meson.current_source_dir() / 'api' / 'test_api.cpp',
]

test_api = executable('quick_dot_ai_test_api',
    test_api_src,
    include_directories: quick_dot_ai_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, quick_dot_ai_layer_dependencies, quick_dot_ai_dep],
    build_by_default: true,
    install: false,
)
