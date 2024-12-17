from openai import OpenAI
from datasets import load_dataset
from jinja2 import Template


def get_examples(dataset_name, val_size):
    # load dataset
    dataset = load_dataset(dataset_name)

    # access the validation dataset
    valid_data = dataset['validation']

    # filter out instances with label -1
    valid_data = valid_data.filter(lambda example: example['label'] != -1)

    # get the first val_size examples from the validation set
    val_examples = valid_data[:val_size]

    # zip the premise and hypothesis lists together to create pairs
    pairs = list(zip(val_examples['premise'], val_examples['hypothesis']))

    formatted_strings = [' </s> '.join(pair) for pair in pairs]
    result = '\n'.join(formatted_strings)

    return result, val_examples['label']


def generate_prompt(text):
    template = Template(

        "Given a sentence pair consisting of a Premise and a Hypothesis, separated by </s>: "
        "Assign one of the following labels to indicate the relationship: \n"
        "Contradiction: The Hypothesis is incompatible with the Premise. "
        "Example: Premise: The sky is sunny. Hypothesis: The sky is stormy. Label: Contradiction \n"
        "Entailment: The Premise logically implies the Hypothesis. The Hypothesis is a logical consequence of the Premise. "
        "Example: Premise: Children waving at camera. Hypothesis: There are children present. Label: Entailment \n"
        "Neutral: There is no clear relationship between the Premise and the Hypothesis. "
        "Example: Premise: The cat is sleeping on the couch. Hypothesis: The shelf is in the room. Label: Neutral \n"
        "The sentence pairs are \n"
        " {{ text }} "
        "Following the examples, the labels are: "
    )
    return template.render(text=text)

def main():
    result, labels= get_examples("snli", 5)
    # print(result)
    prompt = generate_prompt(result)

    client = OpenAI(api_key="api")

    response = client.chat.completions.create(
        model=GPT4All(),
        # model="gpt-4-1106-preview", 
        temperature=0.8,
        messages=[{"role": "user", "content":
                prompt}])
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
