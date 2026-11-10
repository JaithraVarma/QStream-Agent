# Qwen3-0.6B prefill and cached decode baseline

RTX 3060 Laptop GPU, FP16 activations and SDPA. The variants ran in separate processes, with stock FP16 spanning recovery sessions. Each case used two warmup trials and five rounds of three timed trials.
All variants used identical prompts and 16 fixed cached decode steps. Each trial started with a new cache. The prefill returned only last-token logits. Tokenization, loading, conversion and compilation were outside timing.

|Batch|Prompt tokens|Variant|Prefill event ms|Prefill input tok/s|Decode event ms/step|Decode output tok/s|Prefill wall ms|Decode wall ms/step|Peak allocated MiB|Peak reserved MiB|
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|32|fp16|30.512|1048.8|28.806|34.7|30.459|28.521|1150.8|1302.0|
|1|32|reconstructed|35.871|892.1|35.129|28.5|35.134|34.533|1150.8|1314.0|
|1|32|packed|35.213|908.7|36.115|27.7|35.290|36.119|783.6|1242.0|
|1|128|fp16|31.876|4015.5|30.434|32.9|32.058|30.162|1162.3|1178.0|
|1|128|reconstructed|31.243|4096.9|30.202|33.1|31.408|30.119|1162.3|1210.0|
|1|128|packed|37.646|3400.1|35.206|28.4|36.223|32.715|795.1|1198.0|
|1|512|fp16|42.948|11921.5|29.794|33.6|42.760|29.348|1230.5|1302.0|
|1|512|reconstructed|42.861|11945.7|31.220|32.0|42.817|31.259|1235.5|1314.0|
|1|512|packed|50.885|10062.0|35.882|27.9|48.734|34.339|856.4|1242.0|
|4|32|fp16|29.897|4281.4|29.956|133.5|28.520|28.372|1169.3|1186.0|
|4|32|reconstructed|40.945|3126.1|39.226|102.0|38.339|36.990|1169.3|1574.0|
|4|32|packed|36.463|3510.4|33.557|119.2|36.141|32.910|801.5|1258.0|
|4|128|fp16|41.772|12257.0|29.744|134.5|41.803|29.593|1231.5|1302.0|
|4|128|reconstructed|42.334|12094.2|38.151|104.8|41.407|37.276|1221.2|1574.0|
|4|128|packed|60.743|8429.0|33.467|119.5|60.410|33.267|849.2|1258.0|
|4|512|fp16|149.807|13670.9|31.095|128.6|149.477|30.979|1421.3|1532.0|
|4|512|reconstructed|150.872|13574.4|35.829|111.6|138.440|32.974|1421.3|1574.0|
|4|512|packed|179.824|11388.9|34.145|117.1|164.974|31.985|1054.5|1258.0|

Decode ms/step measures one forward for the whole batch. Decode tokens/s counts one output position per sequence per step. Fixed token input avoids sampling cost and sequence divergence. Event windows include idle gaps within the recorded stream. Wall times include the final synchronization for each phase. These are model-forward measurements rather than client time to first token.

The variants ran in separate processes on an unlocked laptop GPU. Raw samples, percentiles, and GPU telemetry are saved alongside the report.

![Timing data](summary.json)

Packed INT4 reduced peak allocated memory by about 367 MiB versus stock FP16 across the six tested workloads. Peak reserved memory did not fall by the same amount. Stock FP16 had lower synchronized wall latency than packed for both phases across the tested grid. Packed decode measured lower latency than reconstructed FP16 in the batch-4 session, while reconstructed was faster at batch 1. Stock and reconstructed both execute FP16 library GEMMs and also differed between sessions, so that split requires a controlled repeat before attributing it to the packed kernel. Token throughput in the table uses event windows, while the plot uses synchronized wall times.
