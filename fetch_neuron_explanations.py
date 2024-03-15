import asyncio
import aiohttp
import pickle
from tqdm import tqdm

'''
Fetch all the OpenAI's neuron explanations of GPT2-small from DeepDecipher API
results are list of tuples like (layer_index, neuron_index, neuron_explanation, score)
[(0, 0, explanation_0, score_0), (0, 1, explanation_1, score_1), ...]
store the results in a pickle file `neuron_explanation_gpt2_small.pkl`
'''

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.create_task(fetch(session, url))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        return responses

def generate_urls():
    urls = []
    for i in range(12):
        for j in range(3072):
            url = f"https://deepdecipher.org/api/gpt2-small/neuron_explainer/{i}/{j}"
            urls.append(url)
    return urls

async def main():
    urls = generate_urls()
    neuron_info_gpt2_small = []
    responses = await fetch_all(urls)
    for response, (i, j) in zip(responses, tqdm([(i, j) for i in range(12) for j in range(3072)])):
        explanation = response['data']['explanation']
        score = response['data']['score']
        neuron_info_gpt2_small.append((i, j, explanation, score))
    
    with open('neuron_explanation_gpt2_small.pkl', 'wb') as file:
        pickle.dump(neuron_info_gpt2_small, file)

# Run the async main function
asyncio.run(main())
