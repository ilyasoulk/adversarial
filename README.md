# adversarial



## Microsoft-phi-1.5 on HumanEval

This repository contains a benchmark of the **Microsoft-phi-1.5** model on the **HumanEval** dataset.
This benchmark supports *pass@k* metrics you can modify the *microsoft_phi/phi.py* file to change the *k* value.
The results are stored in the *results* folder. The *results* folder contains all the generated code and a file *test_results.json* that contains the results of the benchmark.

To run the benchmark you can use the provided Dockerfile. The Dockerfile contains the necessary dependencies to run the benchmark. You can build the Docker image using the following command:
You can find a sample of the results in the *results* folder. (10 first tests of HumanEval on pass@5)

```bash
docker build -t benchmark .
docker run -v $(pwd)/results:/usr/src/app/generated_outputs benchmark
```

## LoRA trainable params on StarCoderBase-1B and Microsoft-phi-2

The *microsoft_phi/lora_params.py* file contains the code to calculate the trainable parameters of the LoRA model for the *StarCoderBase-1B* and *Microsoft-phi-2* models.
Same as earlier you can use the Dockerfile to run the code. Remember to set your own token in the Dockerfile and uncomment the entrypoint.
Feel free to try different models but be careful some models are very large.


For instance, here are the results for these models :

    - StarCoderBase-1B : *trainable params: 835,584 || all params: 1,138,042,880 || trainable%: 0.0734228924660554*
	- Microsoft-phi-2 : *trainable params: 9,175,040 || all params: 2,788,858,880 || trainable%: 0.32898903798244533*

```bash
docker build -t lora_params .
docker run -it lora_params
```


