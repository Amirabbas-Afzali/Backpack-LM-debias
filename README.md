# Model Card for Backpack-GPT2

<!-- Provide a quick summary of what the model is/does. [Optional] -->
The Backpack-GPT2 language model is an instance of the [Backpack architecture](https://arxiv.org/abs/2305.16765), intended to combine strong modeling performance with an interface for interpretability and control.
Most details about this model and its training should be accessed in the paper, [Backpack Language Models](https://arxiv.org/abs/2305.16765).

See also [backpackmodels.science](backpackmodels.science).

![A depiction of the Backpack language modeling process, in which each word in the sequence is weighted and summed to predict each word in context.](http://backpackmodels.science/assets/backpack-process.gif)

# Backpack-LM-debias



# Rax: Learning-to-Rank using JAX 
Rax is a powerful library built on top of JAX for implementing Learning-to-Rank (LTR) models, which are essential for search engines, recommendation systems, and any domain where the ranking of items is critical. By leveraging the speed and automatic differentiation capabilities of JAX, Rax allows for efficient and scalable ranking model development.

# Key Features

JAX Integration: Utilizes JAX for automatic differentiation and accelerated computation, ensuring high performance on both CPUs and GPUs.

Flexible Model Design: Supports a variety of ranking loss functions and metrics, making it easy to tailor models to specific ranking tasks.

Scalability: Designed to handle large datasets and complex models, offering scalability for production-ready applications.

Customizability: Allows customization of ranking models and loss functions, enabling experimentation with novel LTR techniques.


# MS MARCO Dataset
The first dataset was a question answering dataset featuring 100,000 real Bing questions and a human generated answer. Since then we released a 1,000,000 question dataset, a natural langauge generation dataset, a passage ranking dataset, keyphrase extraction dataset, crawling dataset, and a conversational search

# NQ Dataset
Natural Questions contains 307K training examples, 8K examples for development, and a further 8K examples for testing.
In the paper, we demonstrate a human upper bound of 87% F1 on the long answer selection task, and 76% on the short answer selection task.
We believe that matching human performance on this task will require significant progress in natural language understanding; we encourage you to help make this happen.
To help spur development in open-domain question answering, we have created the Natural Questions (NQ) corpus, along with a challenge website based on this data. The NQ corpus contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question. The inclusion of real user questions, and the requirement that solutions should read an entire page to find the answer, cause NQ to be a more realistic and challenging task than prior QA datasets.


