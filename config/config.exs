import Config
# Something is setting this to IEx.Pry so we're overriding it for now. Remove
# if you need to do real debugging
config :elixir, :dbg_callback, {Macro, :dbg, []}

config :ortex,
  add_backend_on_inspect: config_env() != :test

ortex_features =
  case :os.type() do
    {:win32, _} -> ["directml"]
    {:unix, :darwin} -> ["coreml"]
    {:unix, _} -> ["cuda", "tensorrt"]
  end

config :ortex, Ortex.Native, features: ortex_features
