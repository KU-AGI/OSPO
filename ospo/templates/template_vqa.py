def get_attribute_vqa_prompt(prompt):

    system_prompt = 'You are an assistant dedicated to transforming a sentence into several questions. You should first divide it into simple concepts and relations, and then provide the corresponding questions. Avoid using pronouns, such as he, she, it, and they.'
    
    user1 = 'A white harp and a rust soup.'
    answer1 = 'Concepts and relations: a white harp, a rust soup; Questions: Is there a white harp? Is there a rust soup?'
    
    user2 = 'A quarter circle lily and a hexagon mirror.'
    answer2 = 'Concepts and relations: a quarter circle lily, a hexagon mirror; Questions: Is there a quarter-circle lily? Is there a hexagon mirror?'
    
    user3 = 'Shiny mop and metal key holder.'
    answer3 = 'Concepts and relations: a shiny mop, a metal key holder; Questions: Is there a shiny mop? Is there a metal key holder?'
    
    user4 = prompt
    answer4 = ''

    conversation = []
    conversation.append({"role": "<|User|>", "content": user1})
    conversation.append({"role": "<|Assistant|>", "content": answer1})
    conversation.append({"role": "<|User|>", "content": user2})
    conversation.append({"role": "<|Assistant|>", "content": answer2})
    conversation.append({"role": "<|User|>", "content": user3})
    conversation.append({"role": "<|Assistant|>", "content": answer3})
    conversation.append({"role": "<|User|>", "content": user4})
    conversation.append({"role": "<|Assistant|>", "content": answer4})

    return system_prompt, conversation


def get_layout_vqa_prompt(prompt):

    system_prompt = 'You are an assistant dedicated to transforming a sentence into several questions. You should first divide it into simple concepts and relations, and then provide the corresponding questions. Avoid using pronouns, such as he, she, it, and they.'
    
    user1 = 'A pancake on the left of a pasta.'
    answer1 = 'Concepts and relations: a pancake, a pasta, a pancake is on the left of a pasta; Questions: Is there a pancake? Is there a pasta? Is a pancake on the left of a pasta?'
    
    user2 = 'A lamp behind a screwdriver.'
    answer2 = 'Concepts and relations: a lamp, a screwdriver, a lamp is behind a screwdriver; Questions: Is there a lamp? Is there a screwdriver? Is a lamp behind a screwdriver?'
    
    user3 = 'Three light bulbs and eight pumpkins.'
    answer3 = 'Concepts and relations: three light bulbs, eight pumpkins; Questions: Are there three light bulbs? Are there eight pumpkins?'

    user4 = prompt
    answer4 = ''

    conversation = []
    conversation.append({"role": "<|User|>", "content": user1})
    conversation.append({"role": "<|Assistant|>", "content": answer1})
    conversation.append({"role": "<|User|>", "content": user2})
    conversation.append({"role": "<|Assistant|>", "content": answer2})
    conversation.append({"role": "<|User|>", "content": user3})
    conversation.append({"role": "<|Assistant|>", "content": answer3})
    conversation.append({"role": "<|User|>", "content": user4})
    conversation.append({"role": "<|Assistant|>", "content": answer4})

    return system_prompt, conversation


def get_nonspatial_complex_vqa_prompt(prompt):

    system_prompt = 'You are an assistant dedicated to transforming a sentence into several questions. You should first divide it into simple concepts and relations, and then provide the corresponding questions. Avoid using pronouns, such as he, she, it, and they.'
    
    user1 = 'A chef is holding a knife and preparing a dish on the stove.'
    answer1 = 'Concepts and relations: a chef, a knife, a dish, the stove, a chef is holding a knife, a chef is preparing a dish; Questions: Is there a chef? Is there a knife? Is there a dish? Is there a stove? Is a chef holding a knife? Is a chef preparing a dish?'
    
    user2 = 'The green teapot is located near the round oak table.'
    answer2 = 'Concepts and relations: a green teapot, a round oak table, the green teapot is near the round oak table, the round oak table is near the green teapot; Questions: Is there a green teapot? Is there a round oak table? Is the green teapot near the round oak table? Is the round oak table near the green teapot?'
    
    user3 = 'The chunky wooden lamp casts a warm glow on the tattered blue curtains.'
    answer3 = 'Concepts and relations: a chunky wooden lamp, a warm glow, tattered blue curtains, a chunky wooden lamp casts a warm glow, the warm glow is on the tattered blue curtains; Questions: Is there a chunky wooden lamp? Is there a warm glow? Are there tattered blue curtains? Is a chunky wooden lamp casting a warm glow? Is the warm glow on the tattered blue curtains?'
    
    user4 = 'The vibrant orange tomato sat atop the crisp green leaf and the juicy red watermelon.'
    answer4 = 'Concepts and relations: a vibrant orange tomato, a crisp green leaf, a juicy red watermelon, a vibrant orange tomato is atop a crisp green leaf, a vibrant orange tomato is atop a juicy red watermelon; Questions: Is there a vibrant orange tomato? Is there a crisp green leaf? Is there a juicy red watermelon? Is the vibrant orange tomato atop the crisp green leaf? Is the vibrant orange tomato atop the juicy red watermelon?'
    
    user5 = prompt
    answer5 = ''

    conversation = []
    conversation.append({"role": "<|User|>", "content": user1})
    conversation.append({"role": "<|Assistant|>", "content": answer1})
    conversation.append({"role": "<|User|>", "content": user2})
    conversation.append({"role": "<|Assistant|>", "content": answer2})
    conversation.append({"role": "<|User|>", "content": user3})
    conversation.append({"role": "<|Assistant|>", "content": answer3})
    conversation.append({"role": "<|User|>", "content": user4})
    conversation.append({"role": "<|Assistant|>", "content": answer4})
    conversation.append({"role": "<|User|>", "content": user5})
    conversation.append({"role": "<|Assistant|>", "content": answer5})

    return system_prompt, conversation


def get_vqa_prompt(category, prompt):
    if category == "attribute":
        system_prompt, conversation = get_attribute_vqa_prompt(prompt)
    elif category == "layout":
        system_prompt, conversation = get_layout_vqa_prompt(prompt)
    elif category in ["non-spatial", "complex"]:
        system_prompt, conversation = get_nonspatial_complex_vqa_prompt(prompt)

    return system_prompt, conversation
