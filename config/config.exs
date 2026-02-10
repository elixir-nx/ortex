import Config
# Something is setting this to IEx.Pry so we're overriding it for now. Remove
# if you need to do real debugging
config :elixir, :dbg_callback, {Macro, :dbg, []}

config :ortex,
  add_backend_on_inspect: config_env() != :test

# Set the cargo feature flags required to use the matching execution provider
# based on the OS we're running on
default_ortex_features =
  case :os.type() do
    {:win32, _} -> ["directml"]
    {:unix, :darwin} -> ["coreml"]
    {:unix, _} -> []
  end

ortex_features =
  case System.get_env("ORTEX_FEATURES") do
    nil -> default_ortex_features
    "" -> []
    features -> String.split(features, ",", trim: true)
  end

config :ortex, Ortex.Native, features: ortex_features
