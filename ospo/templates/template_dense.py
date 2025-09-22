def get_attribute_system_prompt(base_prompt, negative_prompt):

    system_prompt = "You are an expert prompt engineer for text-to-image models. Your job is to take short and vague prompts and expand them into detailed, descriptive, and unambiguous prompts suitable for high-quality image generation.\
Focus on using full sentences and include visual attributes such as objects, colors, texture, shape.\
Avoid abstract or subjective words and instead use concrete and visual language.\
Do not invent unrelated concepts; Only expand and clarify the given prompt.\
Follow these steps:\n\
Step 1. Extract all objects and their visual attributes from Prompt 1.\
Step 2. For Prompt 1, write a long, rich description that includes all identified objects and attributes from Step 1.\
Step 3. Extract all objects and their visual attributes from Prompt 2.\
Step 4. For Prompt 2, write a long, rich description that includes all identified objects and attributes from Step 3. \
Ensure both outputs share a similar global context or scene.\
"

    final_prompt = f"Prompt 1: {base_prompt}\nPrompt 2: {negative_prompt}\nGenerate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different object-attribute bindings. Let's think step by step."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Prompt 1: A large watermelon\nPrompt 2: A small watermelon\nGenerate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different object-attribute bindings. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: watermelon-['large']\nStep 2. Prompt 1 Dense: A large, ripe watermelon with deep green rinds and faint striping rests heavily in a handwoven wicker basket placed on the grass under dappled sunlight beside a weathered garden shed.\nStep 3. Prompt 2 Object Bindings: watermelon-['small']\nStep 4. Prompt 2 Dense: A small, round watermelon with bright green skin and subtle mottling sits neatly in a handwoven wicker basket placed on the grass under dappled sunlight beside a weathered garden shed."
        },
        {
            "role": "<|User|>",
            "content": "Prompt 1: A peach tree and a square computer keyboard\nPrompt 2: An orange tree and a square keyboard\nGenerate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different object-attribute bindings. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: tree-['peach']; keyboard-['square', 'computer']\nStep 2. Prompt 1 Dense: A peach tree in full bloom stands beside a sleek, square-shaped computer keyboard resting on a polished wooden desk, with soft morning sunlight streaming through the windows of a quiet home office.\nStep 3. Prompt 2 Object Bindings: tree-['orange']; keyboard-['square']\nStep 4. Prompt 2 Dense: An orange tree with lush green leaves and dangling fruit stands beside a sleek, square-shaped computer keyboard resting on a polished wooden desk, with soft morning sunlight streaming through the windows of a quiet home office."
        },
        {
            "role": "<|User|>",
            "content": "Prompt 1: A long in length elephant and a teal owl\nPrompt 2: A short elephant and a blue owl\nGenerate dense, detailed prompts. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: elephant-['long in length']; owl-['teal']\nStep 2. Prompt 1 Dense: A long-bodied elephant with weathered gray skin trudging slowly through a misty jungle path, accompanied by a serene teal owl perched on a mossy branch overhead, its feathers glimmering in the filtered morning light.\nStep 3. Prompt 2 Object Bindings: elephant-['short']; owl-['blue']\nStep 4. Prompt 2 Dense: A short, stocky elephant with wrinkled gray skin navigating a misty jungle path, accompanied by a curious blue owl perched on a mossy branch overhead, its feathers catching flecks of golden morning light."
        },
        {
            "role": "<|User|>",
            "content": final_prompt
        },
        {"role": "<|Assistant|>", "content": ''}
    ]

    return system_prompt, conversation



def get_complex_system_prompt(base_prompt, negative_prompt):

    system_prompt = "You are an expert prompt engineer for text-to-image models. Your job is to take short and vague prompts and expand them into detailed, descriptive, and unambiguous prompts suitable for high-quality image generation.\
Focus on using full sentences and include visual attributes such as objects, colors, texture, shape.\
Avoid abstract or subjective words and instead use concrete and visual language.\
Do not invent unrelated concepts; Only expand and clarify the given prompt.\
Follow these steps:\n\
Step 1. Extract all objects and their visual attributes from Prompt 1.\
Step 2. For Prompt 1, write a long, rich description that includes all identified objects and attributes from Step 1.\
Step 3. Extract all objects and their visual attributes from Prompt 2.\
Step 4. For Prompt 2, write a long, rich description that includes all identified objects and attributes from Step 3. \
Ensure both outputs share a similar global context or scene.\
"
        
    final_prompt = f"Prompt 1: {base_prompt}\nPrompt 2: {negative_prompt}\nGenerate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different object-attribute bindings. Let's think step by step."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Prompt 1: on top of the dark blue, fluffy cloud nestled a brightly shining crescent moon and surrounded it were two towering, crystal clear trees reaching out to meet an invisible bridge made of wispy, ethereal mist.\nPrompt 2: the dark gray, fluffy cloud nestled a brightly glowing crescent moon and surrounded it were two towering, crystal clear trees reaching out to meet an invisible wispy, ethereal bridge made of wispy, ethereal mist.\nGenerate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different object-attribute bindings. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: cloud-['dark blue', 'fluffy'], moon-['brightly shining', 'crescent'], trees-['towering', 'crystal clear'], bridge-['invisible', 'wispy', 'ethereal']\nStep 2. Prompt 1 Dense: On top of a dark blue, fluffy cloud nestled a brightly shining crescent moon, while two towering, crystal clear trees surrounded it, reaching out to connect with an invisible bridge made of wispy, ethereal mist.\nStep 3. Prompt 2 Object Bindings: cloud-['dark gray', 'fluffy'], moon-['brightly glowing', 'crescent'], trees-['towering', 'crystal clear'], bridge-['invisible', 'wispy', 'ethereal']\nStep 4. Prompt 2 Dense: On top of a dark gray, fluffy cloud nestled a brightly glowing crescent moon, while two towering, crystal clear trees surrounded it, reaching out to connect with an invisible bridge made of wispy, ethereal mist."
        },
        {
            "role": "<|User|>",
            "content": "Prompt 1: the vibrant, multi-colored roses wove through the lush, green ivy and gently brushed against the elegant, blue-grey stone fountain, their delicate petals shimmering in the soft breeze.\nPrompt 2: the green, lush ivy wove through the elegant, blue-grey stone fountain, its delicate petals shimmering in the soft breeze.\nGenerate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different object-attribute bindings. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: roses-['vibrant', 'multi-colored'], ivy-['lush', 'green'], fountain-['elegant', 'blue-grey', 'stone'], petals-['delicate', 'shimmering']\nStep 2. Prompt 1 Dense: The vibrant, multi-colored roses wove gracefully through the lush, green ivy and gently brushed against the elegant, blue-grey stone fountain, their delicate petals shimmering in the soft breeze.\nStep 3. Prompt 2 Object Bindings: ivy-['green', 'lush'], fountain-['elegant', 'blue-grey', 'stone'], petals-['delicate', 'shimmering']\nStep 4. Prompt 2 Dense: The green, lush ivy wove gracefully through the elegant, blue-grey stone fountain, its delicate petals shimmering in the soft breeze."
        },
        {
            "role": "<|User|>",
            "content": "Prompt 1: the vibrant orange flowers sprouted on the tall, bare green stalks next to the lush, leafy branches of the big ancient oak tree.\nPrompt 2: the short, leafy green stalks sprouted vibrant yellow flowers next to the small, ancient leafless oak tree.\nGenerate dense, detailed prompts. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: flowers-['vibrant', 'orange'], stalks-['tall', 'bare', 'green'], oak tree-['big', 'ancient', 'lush', 'leafy']\nStep 2. Prompt 1 Dense: The vibrant orange flowers sprouted atop tall, bare green stalks, standing proudly next to the lush, leafy branches of a big, ancient oak tree.\nStep 3. Prompt 2 Object Bindings: flowers-['vibrant', 'yellow'], stalks-['short', 'leafy', 'green'], oak tree-['small', 'ancient', 'leafless']\nStep 4. Prompt 2 Dense: The vibrant yellow flowers sprouted from short, leafy green stalks, positioned beside the small, ancient, leafless oak tree."
        },
        {
            "role": "<|User|>",
            "content": final_prompt
        },
        {"role": "<|Assistant|>", "content": ''}
    ]

    return system_prompt, conversation


def get_spatial_system_prompt(base_prompt, negative_prompt):

    system_prompt = "You are an expert prompt engineer for text-to-image models. Your job is to take short and vague prompts and expand them into detailed, descriptive, and unambiguous prompts suitable for high-quality image generation.\
Focus on using full sentences and include visual attributes such as objects and 2d,3d spatial relations.\
Avoid abstract or subjective words and instead use concrete and visual language.\
Do not invent unrelated concepts; Only expand and clarify the given prompt.\
Follow these steps:\n\
Step 1. Extract all objects and their visual attributes from Prompt 1.\
Step 2. For Prompt 1, write a long, rich description that includes all identified objects and attributes from Step 1.\
Step 3. Extract all objects and their visual attributes from Prompt 2.\
Step 4. For Prompt 2, write a long, rich description that includes all identified objects and attributes from Step 3. \
Ensure both outputs share a similar global context or scene.\
"

    final_prompt = f"Prompt 1: {base_prompt}\nPrompt 2: {negative_prompt}\nGenerate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different object-attribute bindings. Let's think step by step."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Prompt 1: A glowing lantern near a straw hat\nPrompt 2: A flickering candle near a straw hat\nGenerate dense, detailed prompts. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: ['glowing lantern', 'near', 'straw]\nStep 2. Prompt 1 Dense: A softly glowing lantern with an aged brass frame sits near a weathered straw hat placed on a wooden floor, casting a warm light that dances gently across the nearby floorboards in a quiet cabin.\nStep 3. Prompt 2 Object Bindings: ['flickering candle', 'near', 'straw hat']\nStep 4. Prompt 2 Dense: A flickering candle in a small ceramic holder burns near a weathered straw hat placed on a wooden floor, its flame swaying subtly and casting delicate shadows inside a quiet cabin."
        },
        {
            "role": "<|User|>",
            "content": "Prompt 1: A tall cactus behind a metal chair\nPrompt 2: A short cactus in front of a metal chair\nGenerate dense, detailed prompts. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: ['tall cactus', 'behind', 'metal chair']\nStep 2. Prompt 1 Dense: A tall green cactus in a terracotta pot stands behind a minimalist metal chair on a sunlit balcony, its spines casting elongated shadows across the concrete floor.\nStep 3. Prompt 2 Object Bindings: ['short cactus', 'in front of', 'metal chair']\nStep 4. Prompt 2 Dense: A short, stubby cactus in a terracotta pot sits in front of a minimalist metal chair on a sunlit balcony, its compact form creating a rounded shadow on the concrete floor."
        },
        {
            "role": "<|User|>",
            "content": "Prompt 1: A pair of red headphones beside a gaming console\nPrompt 2: A pair of black headphones beneath a TV\nGenerate dense, detailed prompts. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: ['red headphones', 'beside', 'gaming console']\nStep 2. Prompt 1 Dense: A bright red pair of over-ear headphones lies beside a sleek black gaming console on a glass media table, with LED lights reflecting faintly off their glossy surfaces.\nStep 3. Prompt 2 Object Bindings: ['black headphones', 'beneath', 'TV']\nStep 4. Prompt 2 Dense: A matte black pair of over-ear headphones rests beneath a wall-mounted TV on a wooden shelf, their coiled cable looped loosely beside them under the glow of the screen."
        },
        {
            "role": "<|User|>",
            "content": final_prompt
        },
        {"role": "<|Assistant|>", "content": ''}
    ]

    return system_prompt, conversation

def get_numeracy_system_prompt(base_prompt, negative_prompt):

    system_prompt = "You are an expert prompt engineer for text-to-image models. Your job is to take short and vague prompts and expand them into detailed, descriptive, and unambiguous prompts suitable for high-quality image generation.\
Focus on using full sentences and include visual attributes such as objects and numeracy.\
Avoid abstract or subjective words and instead use concrete and visual language.\
Do not invent unrelated concepts; Only expand and clarify the given prompt.\
Follow these steps:\n\
Step 1. Extract all objects and their visual attributes from Prompt 1.\
Step 2. For Prompt 1, write a long, rich description that includes all identified objects and attributes from Step 1.\
Step 3. Extract all objects and their visual attributes from Prompt 2.\
Step 4. For Prompt 2, write a long, rich description that includes all identified objects and attributes from Step 3. \
Ensure both outputs share a similar global context or scene.\
"
    

    final_prompt = f"Prompt 1: {base_prompt}\nPrompt 2: {negative_prompt}\nGenerate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different object-attribute bindings. Let's think step by step."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Prompt 1: Two chairs and three lamps. Prompt 2: Four chairs and one lamp. Generate dense, detailed prompts. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: ['two', 'chairs']; ['three', 'lamps']\nStep 2. Prompt 1 Dense: Two mid-century wooden chairs with curved backs are arranged near a coffee table, while three brass floor lamps cast warm pools of light around the cozy room.\nStep 3. Prompt 2 Object Bindings: ['four chairs', 'one lamp']\nStep 4. Prompt 2 Dense: Four sleek wooden chairs are evenly spaced around a coffee table, with a single tall brass floor lamp casting a gentle glow from the corner of the room."
        },
        {
            "role": "<|User|>",
            "content": "Prompt 1: Two chairs and three lamps\nPrompt 2: Four chairs and one lamp\nGenerate dense, detailed prompts. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: ['two', 'chairs']; ['three', 'lamps']\nStep 2. Prompt 1 Dense: Two mid-century wooden chairs with curved backs are arranged near a coffee table, while three brass floor lamps cast warm pools of light around the cozy room.\nStep 3. Prompt 2 Object Bindings: ['four chairs', 'one lamp']\nStep 4. Prompt 2 Dense: Four sleek wooden chairs are evenly spaced around a coffee table, with a single tall brass floor lamp casting a gentle glow from the corner of the room."
        },
        {
            "role": "<|User|>",
            "content": "Prompt 1: Two apples and four oranges. Prompt 2: Five bananas and one pear. Generate dense, detailed prompts. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: ['two', 'apples']; ['four', 'oranges']\nStep 2. Prompt 1 Dense: Two shiny red apples lie beside four plump oranges in a shallow wicker basket placed on a sunny kitchen counter.\nStep 3. Prompt 2 Object Bindings: ['five bananas', 'one pear']\nStep 4. Prompt 2 Dense: Five curved yellow bananas are stacked casually beside a single ripe green pear on the same wicker basket atop a bright kitchen counter."
        },

        {
            "role": "<|User|>",
            "content": final_prompt
        },
        {"role": "<|Assistant|>", "content": ''}
    ]

    return system_prompt, conversation


def get_non_spatial_system_prompt(base_prompt, negative_prompt):

    system_prompt = "You are an expert prompt engineer for text-to-image models. Your job is to take short and vague prompts and expand them into detailed, descriptive, and unambiguous prompts suitable for high-quality image generation.\
Focus on using full sentences and include visual attributes such as actions.\
Avoid abstract or subjective words and instead use concrete and visual language.\
Do not invent unrelated concepts; Only expand and clarify the given prompt.\
Follow these steps:\n\
Step 1. Extract all objects and their visual attributes from Prompt 1.\
Step 2. For Prompt 1, write a long, rich description that includes all identified objects and attributes from Step 1.\
Step 3. Extract all objects and their visual attributes from Prompt 2.\
Step 4. For Prompt 2, write a long, rich description that includes all identified objects and attributes from Step 3. \
Ensure both outputs share a similar global context or scene.\
"
    
    final_prompt = f"Prompt 1: {base_prompt}\nPrompt 2: {negative_prompt}\nGenerate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different object-attribute bindings. Let's think step by step."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Prompt 1: A man is standing by the kitchen counter, pouring coffee into a mug while listening to the radio. Prompt 2: A man is leaning against the kitchen counter, slicing vegetables while watching a recipe video on his tablet. Generate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different action-related (non-spatial) bindings. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: man-['standing', 'pouring coffee', 'listening to radio']; counter-['kitchen']\n"
                       "Step 2. Prompt 1 Dense: A man stands by a clean kitchen counter, gently pouring steaming coffee into a ceramic mug while a small radio on the windowsill plays soft morning tunes. Sunlight filters through the blinds, casting patterns across the tiled backsplash.\n"
                       "Step 3. Prompt 2 Object Bindings: man-['leaning', 'slicing vegetables', 'watching tablet']; counter-['kitchen']\n"
                       "Step 4. Prompt 2 Dense: A man leans casually against a clean kitchen counter, slicing fresh vegetables with focus while a tablet nearby plays a step-by-step cooking video. The same soft morning light pours through the window, warming the space."
        },
    
        {
            "role": "<|User|>",
            "content": "Prompt 1: A young girl is lying on her bed, sketching animals in her notebook with colored pencils. Prompt 2: A young girl is sitting on her bed, listening to music with headphones while flipping through a magazine. Generate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different action-related (non-spatial) bindings. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: girl-['lying on bed', 'sketching animals']; notebook-['colored pencils']\n"
                       "Step 2. Prompt 1 Dense: A young girl lies sprawled across a neatly made bed in a soft-lit bedroom, focused on sketching cartoon animals in a spiral-bound notebook with a rainbow set of colored pencils scattered beside her.\n"
                       "Step 3. Prompt 2 Object Bindings: girl-['sitting on bed', 'listening to music', 'flipping through magazine']\n"
                       "Step 4. Prompt 2 Dense: A young girl sits cross-legged on the same bed, flipping slowly through a glossy magazine while soft music streams through her large over-ear headphones, the afternoon light warming her cheeks."
        },
        {
        "role": "<|User|>",
        "content": "Prompt 1: A child is crouched in the garden, digging into the soil with a small trowel. Prompt 2: A child is crouched in the garden, observing ants crawling across a rock with great fascination. Generate dense, detailed prompts. Ensure both outputs share a similar global context or scene but have different action-related (non-spatial) bindings. Let's think step by step."
        },
        {
            "role": "<|Assistant|>",
            "content": "Step 1. Prompt 1 Object Bindings: child-['crouched', 'digging soil']; tool-['small trowel']; garden\n"
                       "Step 2. Prompt 1 Dense: A child crouches low in a sunny backyard garden, using a small blue trowel to dig carefully into the soft soil, their sleeves rolled up and cheeks dusted with earth.\n"
                       "Step 3. Prompt 2 Object Bindings: child-['crouched', 'observing ants']; rock; garden\n"
                       "Step 4. Prompt 2 Dense: A child crouches in the same garden, completely absorbed in watching a trail of ants move across a mossy rock, their eyes wide with curiosity as they follow each tiny movement."
        },
        {
            "role": "<|User|>",
            "content": final_prompt
        },
        {"role": "<|Assistant|>", "content": ''}
    ]

    return system_prompt, conversation

# according to sub_category
get_prompt_dense = { 
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