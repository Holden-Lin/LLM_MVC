import PyPDF2

pdf_path =''

# open the file in read-binary mode
with open(pdf_path, 'rb') as file:
    pdf = PyPDF2.PdfFileReader(file)
    text = ''
    # extract texts from each page
    for i in range(pdf.getNumPages()):
        text += pdf.getPage(i).extractText()



#### vetorize data to store in vector DB

import textwrap

# split text into chunks of 100 words each
# TODO: 这里有个超参数
chunks = textwrap.wrap(text,600)
print("check chunks:\n",chunks[:2])


import tensorflow_hub as hub
import hnswlib,numpy

# TODO: 超参数：encoder的对比
model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
vectors = model(chunks)
print("vectorization done")

# initialize the index as before
# dim: dimension of the vectors - TODO:超参数？还是取决于embedding
p = hnswlib.Index(space = 'l2', dim = 512)

# a larger list (higher ef_construction) means the algorithm may examine more candidates and find a closer neighbor but will take more time.
# A higher M makes the graph denser. This can result in higher search speed and accuracy but will slow down the index creation and increase memory usage.
p.init_index(max_elements = len(chunks), ef_construction = 200, M = 16)

# add vectors to the index
p.add_items(vectors)

# create a mapping from index to text
index_to_text = {i: chunk for i, chunk in enumerate(chunks)}
print("indexing done")

#### process user prompts
import openai
openai.api_key = ""

while True:
    
    user_input = input("Enter your query: ")

    # transform user input into a vector using USE
    query_vector = model([user_input])

    # query the index
    labels, distances = p.knn_query(query_vector, k = 5)

    # look up the corresponding chunks of text
    results = [index_to_text[i] for i in labels[0]]
    print("got the VDB indexes:",results)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "You need to provide a comprehensive and coherent answer with reference to the additional data provided. \
            The provided data maybe not all relevant to the user's request and you need to use your fine judgement on whether to refer to it or not.\
            Do you understand the task?"},
            {"role": "assistant", "content": "Yes.Please go ahead."},
            {"role": "user", "content": user_input + " and the addtional data is: /n"+ str(results)}
        ]
    )

    print(completion.choices[0].message)



