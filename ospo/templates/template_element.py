def get_color1_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using color words such as red, orange, yellow, green, blue, navy, purple."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using color words. Except for the color generated before, do not use the same color.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a blue bench.;Concepts and relations: blue bench;Relation: (bench, blue)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using color words. Except for the color generated before, do not use the same color.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a red elephant.;Concepts and relations: red elephant;Relation: (elepahnt, red)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using color words. Except for the color generated before, do not use the same color.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a yellow pendant.;Concepts and relations: yellow pendant;Relation: (pendant, yellow)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using color words. Except for the color generated before, do not use the same color.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation

def get_color2_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using color words such as red, orange, yellow, green, blue, navy, purple."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using color words. Except for the color generated before, do not use the same color.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a blue bench and a green car.;Concepts and relations: blue bench, green car;Relation: (bench, blue), (car, green)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using color words. Except for the color generated before, do not use the same color.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a red elephant and a white grass.;Concepts and relations: red elephant, white grass;Relation: (elepahnt, red), (grass, white)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using color words. Except for the color generated before, do not use the same color.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a yellow pendant and a orange apple;Concepts and relations: yellow pendant, orange apple;Relation: (pendant, yellow), (apple, orange)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using color words. Except for the color generated before, do not use the same color.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation

def get_shape1_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using shape words"

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using shape words. Except for the shape generated before, do not use the same shape.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a diamond tiara.;Concepts and relations: diamond tiara;Relation: (tiara, diamond)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using shape words. Except for the shape generated before, do not use the same shape.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a big elephant.;Concepts and relations: big elephant;Relation: (elepahnt, big)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using shape words. Except for the shape generated before, do not use the same shape.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a circular ceiling fan.;Concepts and relations: circular ceiling fan;Relation: (ceiling fan, circular)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using shape words. Except for the shape generated before, do not use the same shape.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation

def get_shape2_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using shape words"

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using shape words. Except for the shape generated before, do not use the same shape.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a diamond tiara and a round bracelet.;Concepts and relations: diamond tiara, round bracelet;Relation: (tiara, diamond), (bracelet, round)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using shape words. Except for the shape generated before, do not use the same shape.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a big elephant and a small spider.;Concepts and relations: big elephant, small spider;Relation: (elepahnt, big), (spider, small)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using shape words. Except for the shape generated before, do not use the same shape.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a circular ceiling fan and a triangular air vent.;Concepts and relations: circular ceiling fan, triangular air vent;Relation: (ceiling fan, circular), (air vent, triangular)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using shape words. Except for the shape generated before, do not use the same shape.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation

def get_texture1_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using texture words."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using texture words. Except for the texture generated before, do not use the same texture.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a plastic toy.;Concepts and relations: plastic toy;Relation: (toy, plastic)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using texture words. Except for the texture generated before, do not use the same texture.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a rubber band.;Concepts and relations: rubber band;Relation: (band, rubber)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using texture words. Except for the texture generated before, do not use the same texture.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a wooden desk.;Concepts and relations: wooden desk;Relation: (desk, wooden)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using texture words. Except for the texture generated before, do not use the same texture.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation

def get_texture2_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using texture words."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using texture words. Except for the texture generated before, do not use the same texture.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a plastic toy and a metallic keychain.;Concepts and relations: plastic toy, metallic keychain;Relation: (toy, plastic), (keychain, metallic)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using texture words. Except for the texture generated before, do not use the same texture.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a rubber band and a fabric bag.;Concepts and relations: rubber band, fabric bag;Relation: (band, rubber), (bag, fabric)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using texture words. Except for the texture generated before, do not use the same texture.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a wooden desk and a leather shoes.;Concepts and relations: wooden desk, leather shoes;Relation: (desk, wooden), (shoes, leather)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using texture words. Except for the texture generated before, do not use the same texture.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation

def get_2D_spatial_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using 2D spatial relation words. such as on the top of, on the left of, on the right of, on side of"

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using 2D spatial relation words. Except for the 2D spatial relation generated before, do not use the same 2D spatial relation.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a balloon on the top of a giraffe.;Concepts and relations: a balloon, a giraffe, a balloon is on the top of a giraffee;Relation: (ballon, on the top of, giraffee)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using 2D spatial relation words. Except for the 2D spatial relation generated before, do not use the same 2D spatial relation.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a fish on side of a painting.;Concepts and relations: a fish, a painting, a fish is on side of a painting;Relation: (fish, on side of, painting)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using 2D spatial relation words. Except for the 2D spatial relation generated before, do not use the same 2D spatial relation.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a painting on the right of a woman.;Concepts and relations: a painting is on the right of a woman;Relation: (painting, on the right of, woman)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using 2D spatial relation words. Except for the 2D spatial relation generated before, do not use the same 2D spatial relation.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation



def get_3D_spatial_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using 3D spatial relation words such as in front of, hidden by, behind."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using 3D spatial relation words. Except for the 3D spatial relation generated before, do not use the same 3D spatial relation.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a sofa in front of a plate.;Concepts and relations: a sofa, a plate, a sofa is in front of a plate;Relation: (sofa, in front of, plate)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using 3D spatial relation words. Except for the 3D spatial relation generated before, do not use the same 3D spatial relation.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a horse hidden by a television.;Concepts and relations: a horse, a television, a horse is hidden by a television;Relation: (television, hidden by, television)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using 3D spatial relation words. Except for the 3D spatial relation generated before, do not use the same 3D spatial relation.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: a fish behind a butterfly.;Concepts and relations: a fish, a butterfly, a fish is behind a butterfly;Relation: (fish, behind, butterfly)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using 3D spatial relation words. Except for the 3D spatial relation generated before, do not use the same 3D spatial relation.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation

def get_numeracy1_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using numeracy words such as one, two, three, four."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using numeracy words. numeracy must be less than eight.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: one vase.;Concepts and relations: one vase;Relation: (vase, one)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using numeracy words. numeracy must be less than eight.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: four goldfish.;Concepts and relations: four goldfish;Relation: (goldfish, four)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using numeracy words. numeracy must be less than eight.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: three backpacks.;Concepts and relations: three backpacks;Relation: (backpacks, three)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using numeracy words. numeracy must be less than eight.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation


def get_numeracy2_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using numeracy words such as one, two, three, four."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using numeracy words. numeracy must be less than eight.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: one vase and three kites.;Concepts and relations: one vase, three kites;Relation: (vase, one), (kites, three)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using numeracy words. numeracy must be less than eight.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: four goldfish and three microphones.;Concepts and relations: four goldfish, three microphones;Relation: (goldfish, four), (microphones, three)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using numeracy words. numeracy must be less than eight.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: three backpacks and three pigs.;Concepts and relations: three backpacks, three pigs;Relation: (backpacks, three), (pigs, three)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using numeracy words. numeracy must be less than eight.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation


def get_non_spatial_prompt():
    system = "You are an assistant dedicated to generating natural prompts that contain subjects and objects by using non-spatial relationship words such as wear, watch, speak, hold, have, run, look at, talk to, jump, play, walk with, stand on, and sit on."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using non-spatial relationship words.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: Two friends are watching a movie together on a large TV screen.;Concepts and relations: two friends, a movie, two friends are watching a movie;Relation: (two friends, watching, a movie)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using non-spatial relationship words.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: Two athletes are running along the beach as the sun sets behind them.;Concepts and relations: two athletes, the beach, two athletes are running along the beach;Relation: (two athletes, running, along the beach)"},
        {
            "role": "<|User|>",
            "content": "Generate a prompt that contains subjects and objects by using non-spatial relationship words.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation


def get_complex_prompt():
    system = "You are an assistant dedicated to generating natural compositional phrases or prompts, containing multiple objects (number ≥ 2) with one or several adjectives from color, shape, and texture descriptions and spatial (left/right/top/bottom/next to/near/on side of) or nonspatial relationships."

    conversation = [
        {
            "role": "<|User|>",
            "content": "Please generate a compositional phrase or sentence containing multiple objects with one or several adjectives and relationships.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: The fluffy white cat sat next to the black leather couch.;Concepts and relations: fluffy white cat, black leather couch, the fluffy white cat sat next to the black leather couch;Relation: (cat, fluffy white), (couch, black leather), (fluffy white cat, sat next to, black leather couch)"},
        {
            "role": "<|User|>",
            "content": "Please generate a compositional phrase or sentence containing multiple objects with one or several adjectives and relationships.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: The sleek black phone rested beside the textured brown leather wallet.;Concepts and relations: sleek black phone, textured brown leather wallet, the sleek black phone rested beside the textured brown leather wallet;Relation: (phone, sleek black), (wallet, textured brown leather), (sleek black phone, rested beside, textured brown leather wallet)"},
        {
            "role": "<|User|>",
            "content": "Please generate a compositional phrase or sentence containing multiple objects with one or several adjectives and relationships.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: The red spherical balloon floated above the striped rectangular kite and the green triangular flag.;Concepts and relations: red spherical balloon, striped rectangular kite, green triangular flag, red spherical balloon floated above the striped rectangular kite, red spherical balloon floated above the green triangular flag;Relation: (balloon, red spherical), (kite, striped rectangular), (flag, green triangular), (red spherical balloon, floated above, striped rectangular kite), (red spherical balloon, floated above, green triangular flag)"},
        {
            "role": "<|User|>",
            "content": "Please generate a compositional phrase or sentence containing multiple objects with one or several adjectives and relationships.",
        },
        {"role": "<|Assistant|>", "content": "Prompt: The golden, sunlit leaves floated softly above the jagged, rust-colored rocks, their delicate, lacy shapes casting playful shadows on the uneven ground.;Concepts and relations: golden sunlit leaves, jagged rust-colored rocks, uneven ground, golden sunlit leaves floated above the jagged rust-colored rocks, delicate lacy shapes cast shadows on uneven ground;Relation: (leaves, golden sunlit), (rocks, jagged rust-colored), (ground, uneven), (golden sunlit leaves, floated above, jagged rust-colored rocks), (delicate lacy shapes, cast shadows on, uneven ground)"},
        {
            "role": "<|User|>",
            "content": "Please generate a compositional phrase or sentence containing multiple objects with one or several adjectives and relationships.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    return system, conversation


def get_prompt_element(category, processor):

    if category == 'color1':
        system, conversation = get_color1_prompt()
    elif category == 'color2':
        system, conversation = get_color2_prompt()
    elif category == 'texture1':
        system, conversation = get_texture1_prompt()
    elif category == 'texture2':
        system, conversation = get_texture2_prompt()
    elif category == 'shape1':
        system, conversation = get_shape1_prompt()
    elif category == 'shape2':
        system, conversation = get_shape2_prompt()
    elif category == '2D_spatial':
        system, conversation = get_2D_spatial_prompt()
    elif category == '3D_spatial':
        system, conversation = get_3D_spatial_prompt()
    elif category == 'numeracy1':
        system, conversation = get_numeracy1_prompt()
    elif category == 'numeracy2':
        system, conversation = get_numeracy2_prompt()
    elif category == 'non-spatial':
        system, conversation = get_non_spatial_prompt()
    elif category == 'complex':
        system, conversation = get_complex_prompt()
    else:
        raise ValueError(f"Invalid category: {category}")
    
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt=system,
    )

    return sft_format


def get_conversation_element(category, processor):

    if category == 'color1':
        system, conversation = get_color1_prompt()
    elif category == 'color2':
        system, conversation = get_color2_prompt()
    elif category == 'texture1':
        system, conversation = get_texture1_prompt()
    elif category == 'texture2':
        system, conversation = get_texture2_prompt()
    elif category == 'shape1':
        system, conversation = get_shape1_prompt()
    elif category == 'shape2':
        system, conversation = get_shape2_prompt()
    elif category == '2D_spatial':
        system, conversation = get_2D_spatial_prompt()
    elif category == '3D_spatial':
        system, conversation = get_3D_spatial_prompt()
    elif category == 'numeracy1':
        system, conversation = get_numeracy1_prompt()
    elif category == 'numeracy2':
        system, conversation = get_numeracy2_prompt()
    elif category == 'non-spatial':
        system, conversation = get_non_spatial_prompt()
    elif category == 'complex':
        system, conversation = get_complex_prompt()
    else:
        raise ValueError(f"Invalid category: {category}")
    
    return system, conversation
