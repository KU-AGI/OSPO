import random


def get_attribute_system_prompt(p_type, question): # p_type = perturbation type
    assert p_type in ["replace", "swap", "drop"], f"Invalid perturbation type: {p_type}"

    system_prompt = "You are a helpful assistant that generates object-level contrastive prompts by following four steps.\nStep 1. Identify all objects in the prompt.\
Step 2. Identify all object-attribute bindings in the prompt.\
Step 3. Apply one of the following transformations to the given prompt: (1) <replace> an attribute or object with another plausible one, (2) <swap> two attributes or objects, or (3) <drop> an attribute or object.\
Step 4. Generate contrastive prompts based on the transformations.\
Always ensure the output has opposing or conflicting semantics compared to the input. Do not just reorder or repeat the input."

    if p_type == "drop":
        final_prompt = f"{question}\nGenerate object-level contrastive prompts by applying the following transformation: <drop>. Avoid changing the original attribute bindings of the objects that are not dropped. Let's think step by step."
    else: # replace, swap
        final_prompt = f"{question}\nGenerate object-level contrastive prompts by applying the following transformation: <{p_type}>. Let's think step by step."

    conversation = [
        {
            "role": "<|User|>",
            "content": "a blue bench and a red car\nGenerate object-level contrastive prompts by applying the following transformation: <replace>. Let's think step by step."
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: bench, car\nStep 2. Attribute Bindings: ['blue', 'bench'], ['red', 'car']\nStep 3. Transformation: <replace>; Transformed Attributes: ['white', 'bench'], ['green', 'car']\nStep 4. Contrastive Prompt: a white bench and a green car"},
        {
            "role": "<|User|>",
            "content": "a rubber tire and a wooden desk\nGenerate object-level contrastive prompts by applying the following transformation: <swap>. Let's think step by step.",
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: tire, desk\nStep 2. Attribute Bindings: ['rubber', 'tire'], ['wooden', 'desk']\nStep 3. Transformation: <swap>; Transformed Attributes: ['wooden', 'tire'], ['rubber', 'desk']\nStep 4. Contrastive Prompt: a wooden tire and a rubber desk"},
        {
            "role": "<|User|>",
            "content": "a oblong sweet potato and teardrop mango\nGenerate object-level contrastive prompts by applying the following transformation: <drop>. Avoid changing the original attribute bindings of the objects that are not dropped. Let's think step by step."
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: sweet potato, mango\nStep 2. Attributes Bindings: ['oblong', 'sweet potato'], ['teardrop', 'mango']\nStep 3. Transformation: <drop>; Transformed Attributes: ['teardrop', 'mango']\nStep 4. Contrastive Prompt: a teardrop mango"},
        {
            "role": "<|User|>",
            "content": final_prompt
        },
        {"role": "<|Assistant|>", "content": ''},
    ]
    return system_prompt, conversation



def get_spatial_system_prompt(p_type, question): # p_type = perturbation type
    assert p_type in ["replace", "swap", "drop"], f"Invalid perturbation type: {p_type}"

    system_prompt = "You are a helpful assistant that generates object-level contrastive prompts by following four steps.\nStep 1. Identify all objects in the prompt.\
Step 2. Identify all object-attribute bindings in the prompt.\
Step 3. Apply one of the following transformations to the given prompt: (1) <replace> a spatial relationship with another plausible one, (2) <swap> two objects placed side by side in the spatial layout, or (3) <drop> a spatial relationship.\
Step 4. Generate contrastive prompts based on the transformations.\
Always ensure the output has opposing or conflicting semantics compared to the input. Do not just reorder or repeat the input."

    # To avoid overfitting to specific examle in <replace> case
    random_spatial = ["under", "next to", "in front of", "behind", "to the left of", "to the right of"]
    selected = random.choice(random_spatial)

    if p_type == "replace":
        final_prompt = f"{question}\nGenerate object-level contrastive prompts by applying the following transformation: <replace>. Avoid just using '{selected}' in the example. Let's think step by step." 
    else: # swap, drop
        final_prompt = f"{question}\nGenerate object-level contrastive prompts by applying the following transformation: <{p_type}>. Let's think step by step."

    conversation = [
        {
            "role": "<|User|>",
            "content": "A xylophone above a computer\nGenerate object-level contrastive prompts by applying the following transformation: <replace>. Let's think step by step."
        },
        {"role": "<|Assistant|>", "content": f"Step 1. Objects: xylophone, computer\nStep 2. Attribute Bindings: ['xylophone', 'above', 'computer']\nStep 3. Transformation: <replace>; Transformed Attributes: ['xylophone', '{selected}', 'computer']\nStep 4. Contrastive Prompt: A xylophone {selected} a computer"},
        {
            "role": "<|User|>",
            "content": "A ramen bowl to the right of a coffee cup\nGenerate object-level contrastive prompts by applying the following transformation: <swap>. Let's think step by step.",
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: ramen bowl, coffee cup\nStep 2. Attribute Bindings: ['ramen bowl', 'the right of', 'coffee cup']\nStep 3. Transformation: <swap>; Transformed Attributes: ['coffee cup', 'the right of', 'ramen bowl']\nStep 4. Contrastive Prompt: A coffee cup to the right of a ramen bowl"},
        {
            "role": "<|User|>",
            "content": "A microwave oven next to a sweatshirt\nGenerate object-level contrastive prompts by applying the following transformation: <drop>. Let's think step by step."
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: microwave oven, sweatshirt\nStep 2. Attributes Bindings: ['microwave oven', 'next to', 'sweatshirt']\nStep 3. Transformation: <drop>; Transformed Attributes: ['microwave oven', 'sweatshirt']\nStep 4. Contrastive Prompt: A microwave oven and a sweatshirt"},
        {
            "role": "<|User|>",
            "content": final_prompt
        },
        {"role": "<|Assistant|>", "content": ''},
    ]
    return system_prompt, conversation



def get_numeracy_system_prompt(p_type, question): # p_type = perturbation type
    assert p_type in ["replace", "swap", "drop"], f"Invalid perturbation type: {p_type}"

    system_prompt = "You are a helpful assistant that generates object-level contrastive prompts by following four steps.\nStep 1. Identify all objects in the prompt.\
Step 2. Identify all object-number bindings in the prompt.\
Step 3. Apply one of the following transformations to the given prompt: (1) <replace> a number with another plausible one, (2) <swap> two numbers, or (3) <drop> a number or object.\
Step 4. Generate contrastive prompts based on the transformations from Step 3.\
Always ensure the output has opposing or conflicting semantics compared to the input."

    if p_type == "replace":
        final_prompt = f"{question}\nGenerate object-level contrastive prompts by applying the following transformation: <replace>. Replace the number with new one. Avoid answering the original input at Step 4. Let's think step by step."
    else: # swap, drop
        final_prompt = f"{question}\nGenerate object-level contrastive prompts by applying the following transformation: <{p_type}>. Avoid answering the original input at Step 4. Let's think step by step."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Three coffee cups and four telephones\nGenerate object-level contrastive prompts by applying the following transformation: <replace>. Replace the number with new one. Avoid answering the original input at Step 4. Let's think step by step.",
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: coffe cups, telephones\nStep 2. Number Bindings: ['three', 'coffee cups'], ['four', 'telephones']\nStep 3. Transformation: <replace>; Transformed Bindings: ['two', 'coffee cups'], ['six', 'telephones']\nStep 4. Contrastive Prompt: two coffee cups and six telephones"},
        {
            "role": "<|User|>",
            "content": "A desk and five rugs\nGenerate object-level contrastive prompts by applying the following transformation: <swap>. Avoid answering the original input at Step 4. Let's think step by step.",
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: desk, rugs\nStep 2. Number Bindings: ['one', 'desk'], ['five', 'rugs']\nStep 3. Transformation: <swap>; Transformed Bindings: ['five', 'desk'], ['one', 'rugs']\nStep 4. Contrastive Prompt: five desks and one rug"},
        
        {
            "role": "<|User|>",
            "content": "Two apples and a blouse\nGenerate object-level contrastive prompts by applying the following transformation: <drop>. Avoid answering the original input at Step 4. Let's think step by step."
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: apples, blouse\nStep 2. Number Bindings: ['two', 'apples'], ['one', 'blouse']\nStep 3. Transformation: <drop>; Transformed Bindings: ['two', 'apples']\nStep 4. Contrastive Prompt: two apples"},
        {
            "role": "<|User|>",
            "content": final_prompt
        },
        {"role": "<|Assistant|>", "content": ''},
    ]
    return system_prompt, conversation
    

def get_non_spatial_system_prompt(p_type, question): # p_type = perturbation type
    assert p_type in ["replace", "swap", "drop"], f"Invalid perturbation type: {p_type}"
    if p_type == "replace":
        system_prompt = "You are a helpful assistant that generates object-level contrastive prompts by following four steps.\nStep 1. Identify all subjects in the prompt.\
Step 2. Identify all subject-action or subject-action-object/complement bindings.\
Step 3. Apply the following transformations to the prompt: <replace> an action, object, or complement with another plausible but semantically distinct one.\
Step 4. Generate a contrastive prompt using the transformed bindings.\
Always ensure the output has opposing or conflicting semantics compared to the input. Avoid just reordering or repeating the input."
        conversation = [
            {
                "role": "<|User|>",
                "content": "a couple is having a loud arguments while watching baseball in the backyard.\nGenerate object-level contrastive prompts by applying the following transformation: <replace>. Do not reuse the same actions, verbs, or semantics. Let's think step by step.",
            },
            {"role": "<|Assistant|>", "content": "Step 1. Subjects: a couple\nStep 2. Action Bindings: ['a couple', 'is having', 'a loud arguments'], ['a couple', 'is watching', 'baseball', 'in the backyard.']\nStep 3. Transformation: <replace>; Transformed Actions: ['a couple', 'is speaking', 'a loud arguments'], ['a couple', 'is playing', 'baseball', 'in the backyard.']\nStep 4. Contrastive Prompt: a couple is speaking a loud arguments while playing baseball in the backyard."},
            {
                "role": "<|User|>",
                "content": "a grandfather shares a funny story with his grandson while sitting on the sofa.\nGenerate object-level contrastive prompts by applying the following transformation: <replace>. Do not reuse the same actions, verbs, or semantics. Let's think step by step."
            },
            {"role": "<|Assistant|>", "content": "Step 1. Subjects: a grandfather, his grandson\nStep 2. Action Bindings: ['a grandfather', 'shares', 'a funny story'], ['grandson', 'is sitting', 'on the sofa']\nStep 3. Transformation: <replace>; Transformed Actions: ['a grandfather', 'reads', 'a funny story'], ['grandson', 'is jumping', 'on the sofa']\nStep 4. Contrastive Prompt: A grandfather reads a funny story with his grandson while jumping on the sofa."},
            {
                "role": "<|User|>",
                "content": "a teacher is having a conversation with two students about their report cards.\nGenerate object-level contrastive prompts by applying the following transformation: <replace>. Do not reuse the same actions, verbs, or semantics. Let's think step by step."
            },
            {"role": "<|Assistant|>", "content": "Step 1. Subjects: a teacher\nStep 2. Action Bindings: ['a teacher', 'is having', 'a conversation' 'with two students']\nStep 3. Transformation: <replace>; Transformed Actions: ['a teacher', 'is grading', 'assignments' 'with two students']\nStep 4. Contrastive Prompt: a teacher is grading assignments with two students about their report cards."},
            {
                "role": "<|User|>",
                "content": f"{question}\nGenerate object-level contrastive prompts by applying the following transformation: <replace>. Do not reuse the same actions, verbs, or semantics. Let's think step by step."
            },
            {"role": "<|Assistant|>", "content": ''},
        ]
    
    elif p_type == "drop":
        system_prompt = "You are a helpful assistant that generates object-level contrastive prompts by applying the <drop> transformation.\n\nYou must follow these steps:\nStep 1. Identify all objects in the prompt.\
Step 2. Randomly select one of the objects from Step 1.\
Step 3. Mark the selected object with the tag <pos> in the original prompt.\
Step 4. Remove the <pos>-tagged object entirely from the prompt to generate the contrastive output.\
Avoid just repeating the input and ensure that the ouput has less specific information than input."
        conversation = [
        {
            "role": "<|User|>",
            "content": "a couple is having a loud arguments while watching baseball in the backyard.\nGenerate object-level contrastive prompts by applying the following transformation: <drop>. Let's think step by step. Avoid including selected object in the output.",
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: ['arguments', 'baseball', 'backyard']\nStep 2. Selected Object: ['backyard']\nStep 3. a couple is having a loud arguments while watching baseball in the <pos>.\nStep 4.Contrastive Prompt: a couple is having a loud arguments while watching baseball in the."},
        {
            "role": "<|User|>",
            "content": "a grandfather shares a funny story with his grandson while sitting on the sofa.\nGenerate object-level contrastive prompts by applying the following transformation: <drop>. Let's think step by step. Avoid including selected object in the output."
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: ['story', 'sofa']\nStep 2. Selected Object: ['story']\nStep 3. a grandfather shares a funny <pos> with his grandson while sitting on the sofa.\nStep 4.Contrastive Prompt: a grandfather shares a funny with his grandson while sitting on the sofa."},
        {
            "role": "<|User|>",
            "content": "a person is writing a recipe down with a tablet while sitting on the couch.\nGenerate object-level contrastive prompts by applying the following transformation: <drop>. Let's think step by step. Avoid including selected object in the output."
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: ['recipe', 'tablet', 'couch']\nStep 2. Selected Object: ['tablet']\nStep 3. a person is writing a recipe down with a <pos> while sitting on the couch.\nStep 4.Contrastive Prompt: a person is writing a recipe down with a while sitting on the couch."},
        {
            "role": "<|User|>",
            "content": f"{question}\nGenerate object-level contrastive prompts by applying the following transformation: <drop>. Let's think step by step. Avoid including selected object in the output." 
        },
        {"role": "<|Assistant|>", "content": ''},
        ]

    else:
        return None, None # NotImplemented
    
    return system_prompt, conversation




def get_complex_system_prompt(p_type, question):
    assert p_type in ["replace", "swap", "drop"], f"Invalid perturbation type: {p_type}"

    system_prompt = "You are a helpful assistant that generates object-level contrastive prompts by following four steps.\nStep 1. Identify all objects in the prompt.\
Step 2. Identify all object-attribute bindings in the prompt.\
Step 3. Apply one of the following transformations to the given prompt: (1) <replace> an attribute with another plausible one, (2) <swap> two attributes or objects, or (3) <drop> an attribute or object.\
Step 4. Generate contrastive prompts based on the transformations.\
Always ensure the output has opposing or conflicting semantics compared to the input."
    
    if p_type == "drop":
        final_prompt = f"{question}\nGenerate object-level contrastive prompts by applying the following transformation: <drop>. Avoid changing the original attribute bindings of the objects that are not dropped. Let's think step by step."
    else: # replace, swap
        final_prompt = f"{question}\nGenerate object-level contrastive prompts by applying the following transformation: <{p_type}>. Let's think step by step."

    conversation = [
        {
            "role": "<|User|>",
            "content": "the gleaming silver, spiral staircase connected every floor, its majestic curves wrapping around the living room like a graceful, snake-like vine.\nGenerate object-level contrastive prompts by applying the following transformation: <swap>. Let's think step by step.",
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: staircase, floor, curves, living room, vine\nStep 2. Attribute Bindings: staircase-['gleaming', 'silver', 'spiral']; curves-['majestic']; vine-['graceful', 'snake-like']\nStep 3. Transformation: <swap>; Transformed Attributes: staircase-['graceful', 'snake-like']; curves-['majestic']; vine-['gleaming', 'silver', 'spiral']\nStep 4. Contrastive Prompt: the graceful, snake-like staircase connected every floor, its majestic curves wrapping around the living room like a gleaming silver, spiral vine."},
        {
            "role": "<|User|>",
            "content": "the tall, slender white tent stood out next to the thick, broad, tan tree.\nGenerate object-level contrastive prompts by applying the following transformation: <replace>. Let's think step by step."
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: tent, tree\nStep 2. Attribute Bindings: tent-['tall', 'slender', 'white']; tree-['thick', 'broad', 'tan']\nStep 3. Transformation: <replace>; Transformed Attributes: tent-['small', 'slender', 'white']; tree-['thick', 'broad', 'yellow']\nStep 4. Contrastive Prompt: the small, slender white tent stood out next to the thick, broad, yellow tree."},
        {
            "role": "<|User|>",
            "content": "the turquoise fish darted gracefully beside the green, moss-covered lily pads floating on calm reflective waters.\nGenerate object-level contrastive prompts by applying the following transformation: <drop>. Avoid changing the original attribute bindings of the objects that are not dropped. Let's think step by step."
        },
        {"role": "<|Assistant|>", "content": "Step 1. Objects: fish, pads, waters\nStep 2. Attributes Bindings: fish-['turquoise']; pads-['green', 'moss-covered', 'lily']; waters-['calm', 'reflective']\nStep 3. Transformation: <drop>; Transformed Attributes: pads-['green', 'moss-covered', 'lily']; waters-['calm', 'reflective']\nStep 4. Contrastive Prompt: the green, moss-covered lily pads floating on calm reflective waters."},
        {
            "role": "<|User|>",
            "content": final_prompt
        },
        {"role": "<|Assistant|>", "content": ''},
    ]
    return system_prompt, conversation
    

# according tos sub_category
get_prompt_negative = { 
            "attribute1_color": get_attribute_system_prompt, 
            "attribute1_texture": get_attribute_system_prompt, 
            "attribute1_shape": get_attribute_system_prompt,
            "attribute2": get_attribute_system_prompt,
            "layout1": get_spatial_system_prompt,
            "layout2": get_numeracy_system_prompt,
            "layout3": get_numeracy_system_prompt,
            "non-spatial": get_non_spatial_system_prompt,
            "complex": get_complex_system_prompt
        }
