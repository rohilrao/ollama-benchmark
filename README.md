# ollama-benchmark
benchmarking scripts for open source model inference using ollama


Task is to create outputs like so:

<img width="1391" height="564" alt="{4CEA40B9-ACDC-42D4-841B-3FBBE051D930}" src="https://github.com/user-attachments/assets/be35b1df-e719-4bbb-936b-ad0c48c69d0f" />



model,n,num_ctx,pad_words,status,error_message,vram_mb,elapsed_time_sec,ps_after
mistral-small3.2:24b-32k,4,16384,0,ok,,181574.3671875,10.406090586446226,
mistral-small3.2:24b-32k,4,32768,0,ok,,501910.37109375,13.152654348872602,
mistral-small3.2:24b-32k,4,65536,0,oom_or_runner_crash,llama runner process has terminated: %!w(<nil>) (status code: 500) | llama runner process has terminated: %!w(<nil>),,242.05019910726696,[]
mistral-small3.2:24b-32k,20,16384,0,ok,,181574.3671875,17.92073257919401,
mistral-small3.2:24b-32k,20,32768,0,ok,,501910.37109375,17.880543034989387,
mistral-small3.2:24b-32k,20,65536,0,oom_or_runner_crash,llama runner process has terminated: %!w(<nil>) (status code: 500) | llama runner process has terminated: %!w(<nil>),,242.05804026313126,[]


model,n,status,error_message,elapsed_time_sec,wall_time_sec,ttft_sec,tokens_per_sec,batch_tokens_per_sec,thinking_tokens,content_tokens,n_successful_repeats,n_failed_repeats
mistral-small3.2:24b-32k,4,ok,,2.8065931820310652,2.770505942811724,0.27339097496587783,44.02165561736795,131.1176442988899,0,91.0,2,0
mistral-small3.2:24b-32k,20,ok,,10.259790496667847,9.54913796283072,0.737444663059432,14.368519375995373,183.28703655934214,0,88.175,2,0

