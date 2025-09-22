BLIPVQA_PROJECT_PATH = "{YOUR_PATH}/T2I-CompBench/BLIPvqa_eval"
UNIDET_PROJECT_PATH = "{YOUR_PATH}/T2I-CompBench/UniDet_eval"
COMPLEX_DATASET_PATH = "{YOUR_PATH}/T2I-CompBench/examples/dataset/"

import os
import sys
import json
import clip
import collections
import argparse
from PIL import Image
import numpy as np
import torch
import spacy
from tqdm import tqdm
from accelerate import Accelerator
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import build_config


# STEP 1
def blipVQA_eval(measure:list, out_dir, img_dir, np_index=8):

    print("\n*** blipVQA Eval Start ***\n")

    project_dir=BLIPVQA_PROJECT_PATH # "eval/t2i/T2I-CompBench/BLIPvqa_eval/"
    os.chdir(project_dir)
    sys.path.append(os.getcwd()) 

    base_out_dir = os.path.join(out_dir, "1_blipVQA_eval")
    base_img_dir = img_dir

    # if `complex` is in measure, complex is placed ALWAYS at the last.
    if "complex" in measure:
        measure.remove("complex")
        measure.append("complex") # re-order

    print_score = collections.defaultdict(float)

    for m in measure: 
        
        answer = []
        img_dir = os.path.join(base_img_dir, m)
        out_dir = os.path.join(base_out_dir, m)

        sample_num = len(os.listdir(img_dir)) 
        reward = torch.zeros((sample_num, np_index)).to(device='cuda')

        # np_index = args.np_num # = how many noun phrases
        for i in tqdm(range(np_index)):
            print(f"start VQA{i+1}/{np_index}!")
            os.makedirs(f"{out_dir}/annotation{i + 1}", exist_ok=True) 
            os.makedirs(f"{out_dir}/annotation{i + 1}/VQA/", exist_ok=True)
            
            from BLIP_vqa import Create_annotation_for_BLIP
            Create_annotation_for_BLIP(
                f"{img_dir}/",                  # image folder
                f"{out_dir}/annotation{i + 1}", # output folder
                np_index=i,
            )

            from BLIP.train_vqa_func import VQA_main
            answer_tmp = VQA_main(f"{out_dir}/annotation{i + 1}/",
                                f"{out_dir}/annotation{i + 1}/VQA/")
            answer.append(answer_tmp)

            with open(f"{out_dir}/annotation{i + 1}/VQA/result/vqa_result.json", "r") as file:
                r = json.load(file)
            with open(f"{out_dir}/annotation{i + 1}/vqa_test.json", "r") as file:
                r_tmp = json.load(file)
            for k in range(len(r)):
                if(r_tmp[k]['question']!=''):
                    reward[k][i] = float(r[k]["answer"])
                else:
                    reward[k][i] = 1
            print(f"end VQA{i+1}/{np_index}!")
        reward_final = reward[:,0]
        for i in range(1,np_index):
            reward_final *= reward[:,i]

        # output final json
        with open(f"{out_dir}/annotation{i + 1}/VQA/result/vqa_result.json", "r") as file:
            r = json.load(file)
        reward_after=0
        for k in range(len(r)):
            r[k]["answer"] = '{:.4f}'.format(reward_final[k].item())
            reward_after+=float(r[k]["answer"])

        os.makedirs(f"{out_dir}/annotation", exist_ok=True)
        with open(f"{out_dir}/annotation/vqa_result.json", "w") as file:
            json.dump(r, file)

        # calculate avg of BLIP-VQA as BLIP-VQA score
        print("BLIP-VQA score:", reward_after/len(r),'!\n')
        with open(f"{out_dir}/annotation/blip_vqa_score.txt", "w") as file:
            file.write("BLIP-VQA score:"+str(reward_after/len(r)))

        # check if it is `complex`
        if m == "complex":
            print("\n*** blipVQA Eval Done ***")
            # return r # = `attribute_score` json 
            return (print_score, r)
        else:
            print_score[m] = reward_after/len(r)

    print(print_score)
    print("\n*** blipVQA Eval Done ***\n")
    return (print_score, r)


# STEP 2
def uniDet_2D_eval(measure:list, out_dir, img_dir):

    print("\n*** uniDet 2D Eval Start ***\n")  
    project_dir=UNIDET_PROJECT_PATH # "eval/t2i/T2I-CompBench/UniDet_eval/"
    os.chdir(project_dir)
    sys.path.append(os.getcwd()) 

    base_out_dir = os.path.join(out_dir, "2_uniDet_2D_eval")
    base_img_dir = img_dir

    from _2D_spatial_eval import determine_position, get_mask_labels
    from experts.model_bank import load_expert_model
    from experts.obj_detection.generate_dataset import Dataset, collate_fn

    model, transform = load_expert_model(task='obj_detection', ckpt="RS200")
    accelerator = Accelerator(mixed_precision='fp16')
    obj_label_map = torch.load('dataset/detection_features.pt')['labels']

    # if `complex` is in measure, complex is placed ALWAYS at the last.
    if "complex" in measure:
        measure.remove("complex")
        measure.append("complex") # re-order

    print_score = collections.defaultdict(float)

    for m in measure:
        img_dir = os.path.join(base_img_dir, m)
        save_path= os.path.join(base_out_dir, m) # f'{out_dir}/labels' # 2d_spatial

        batch_size = 64
        dataset = Dataset(img_dir,  transform)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            # pin_memory=True,
            collate_fn=collate_fn,
        )

        model, data_loader = accelerator.prepare(model, data_loader)

        with torch.no_grad():
            result = []
            map_result = []
            for i, test_data in enumerate(tqdm(data_loader)):
                test_pred = model(test_data)
                for k in range(len(test_pred)):
                    instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor  # get the bbox of list
                    instance_id = test_pred[k]['instances'].get_fields()['pred_classes']
                    depth = test_data[k]['image'][0]

                    # get score
                    instance_score = test_pred[k]['instances'].get_fields()['scores']

                    obj_bounding_box, obj_labels_dict = get_mask_labels(depth, instance_boxes, instance_id)

                    obj = []  
                    for i in range(len(obj_bounding_box)):
                        obj_name = obj_label_map[obj_labels_dict[i]]  
                        obj.append(obj_name)

                    img_path_split = test_data[k]['image_path'].split('/')
                    prompt = img_path_split[-1].split('_')[0] # get prompt from file names
                    vocab_spatial = ['on side of', 'next to', 'near', 'on the left of', 'on the right of', 'on the bottom of', 'on the top of','on top of'] #locality words

                    locality = None
                    for word in vocab_spatial:
                        if word in prompt:
                            locality = word
                            break

                    if m=="complex": # if (args.complex)
                        #for complex structure
                        nlp = spacy.load('en_core_web_sm')
                        # Define the sentence
                        sentence = prompt
                        # Process the sentence using spaCy
                        doc = nlp(sentence)
                        # Define the target prepositions
                        prepositions = ["on top of", "on bottom of", "on the left", "on the right",'next to','on side of','near']
                        # Extract objects before and after the prepositions
                        objects = []
                        for i in range(len(doc)):
                            if doc[i:i + 3].text in prepositions or doc[i:i + 2].text in prepositions or doc[i:i + 1].text in prepositions:
                                if doc[i:i + 3].text in prepositions:
                                    k=3
                                elif doc[i:i + 2].text in prepositions:
                                    k=2
                                elif doc[i:i + 1].text in prepositions:
                                    k=1
                                preposition_phrase = doc[i:i + 3].text
                                for j in range(i - 1, -1, -1):
                                    if doc[j].pos_ == 'NOUN':
                                        objects.append(doc[j].text)
                                        break
                                    elif doc[j].pos_ == 'PROPN':
                                        objects.append(doc[j].text)
                                        break
                                flag=False
                                for j in range(i + k, len(doc)):
                                    if doc[j].pos_ == 'NOUN':
                                        objects.append(doc[j].text)
                                        break
                                    if(j==len(doc)-1):
                                        flag=True 
                                if flag:
                                    for j in range(i + k, len(doc)):
                                        if (j+1<len(doc)) and doc[j].pos_ == 'PROPN' and doc[j+1].pos_ != 'PROPN':
                                            objects.append(doc[j].text)
                                            break
                        if (len(objects)==2):
                            obj1=objects[0]
                            obj2=objects[1]
                        else:
                            obj1=None
                            obj2=None
                    else:
                        #for simple structure
                        nlp = spacy.load("en_core_web_sm")
                        doc = nlp(prompt)
                        obj1= [token.text for token in doc if token.pos_=='NOUN'][0]
                        obj2= [token.text for token in doc if token.pos_=='NOUN'][-1]

                    person = ['girl','boy','man','woman']
                    if obj1 in person:
                        obj1 = "person"
                    if obj2 in person:
                        obj2 = "person"
                    if obj1 in obj and obj2 in obj:
                        obj1_pos = obj.index(obj1)
                        obj2_pos = obj.index(obj2)
                        obj1_bb = obj_bounding_box[obj1_pos]
                        obj2_bb = obj_bounding_box[obj2_pos]
                        box1, box2={},{}

                        box1["x_min"] = obj1_bb[0]
                        box1["y_min"] = obj1_bb[1]
                        box1["x_max"] = obj1_bb[2]
                        box1["y_max"] = obj1_bb[3]
                        box2["x_min"] = obj2_bb[0]
                        box2["y_min"] = obj2_bb[1]
                        box2["x_max"] = obj2_bb[2]
                        box2["y_max"] = obj2_bb[3]


                        score = 0.25 * instance_score[obj1_pos].item() + 0.25 * instance_score[obj2_pos].item()  # score = avg across two objects score
                        score += determine_position(locality, box1, box2) / 2
                    elif obj1 in obj:
                        obj1_pos = obj.index(obj1)  
                        score = 0.25 * instance_score[obj1_pos].item()
                    elif obj2 in obj:
                        obj2_pos = obj.index(obj2)
                        score = 0.25 * instance_score[obj2_pos].item()
                    else:
                        score = 0
                    if (score<0.5):
                        score=0

                    image_dict = {}
                    image_dict['question_id']=int(img_path_split[-1].split('_')[-1].split('.')[0])
                    image_dict['answer'] = score
                    result.append(image_dict)

                    # add mapping
                    map_dict = {}
                    map_dict['image'] = img_path_split[-1]
                    map_dict['question_id']=int(img_path_split[-1].split('_')[-1].split('.')[0])
                    map_result.append(map_dict)
            

            im_save_path = os.path.join(save_path, 'annotation_obj_detection_2d')
            os.makedirs(im_save_path, exist_ok=True)

            with open(os.path.join(im_save_path, 'vqa_result.json'), 'w') as f:
                json.dump(result, f)
            print('vqa result saved in {}'.format(im_save_path))

            # avg score
            avg_score = 0
            for i in range(len(result)):
                avg_score+=float(result[i]['answer'])
            with open(os.path.join(im_save_path, 'avg_score.txt'), 'w') as f:
                f.write('score avg:'+str(avg_score/len(result)))
            print("avg score:",avg_score/len(result))
            

            # save mapping
            with open(os.path.join(im_save_path, 'mapping.json'), 'w') as f:
                json.dump(map_result, f)

        # check if it is `complex`
        if m == "complex":
            print("\n*** uniDet 2D Eval Done ***")
            # return result # = `spatial_score` json 
            return (print_score, result)
        else:
            print_score[m] = avg_score/len(result)
    
    print(print_score)
    print("\n*** uniDet 2D Eval Done ***\n")  
    return (print_score, result)      


# STEP 3
def clipScore_eval(measure:list, out_dir, img_dir):

    print("\n*** CLIPscore Eval Start ***\n")

    base_out_dir = os.path.join(out_dir, "3_clipScore_eval")
    base_img_dir = img_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    nlp=spacy.load('en_core_web_sm')

    # if `complex` is in measure, complex is placed ALWAYS at the last.
    if "complex" in measure:
        measure.remove("complex")
        measure.append("complex") # re-order

    print_score = collections.defaultdict(float)

    for m in measure:
        img_dir = os.path.join(base_img_dir, m)
        out_dir= os.path.join(base_out_dir, m) 

        file_names = os.listdir(img_dir)
        file_names.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))  # sort

        cnt = 0
        total = []

        # output annotation.json
        for file_name in file_names:

            image_path = os.path.join(img_dir, file_name)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            prompt = file_name.split("_")[0]

            if m=="complex": # if (args.complex) 
                doc=nlp(prompt)
                prompt_without_adj=' '.join([token.text for token in doc if token.pos_ != 'ADJ']) #remove adj
                text = clip.tokenize(prompt_without_adj).to(device)
            else:
                text = clip.tokenize(prompt).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image.to(device))
                image_features /= image_features.norm(dim=-1, keepdim=True)

                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Calculate the cosine similarity between the image and text features
                cosine_similarity = (image_features @ text_features.T).squeeze().item()

            similarity = cosine_similarity
            cnt+=1
            # if (cnt % 100 == 0):
                # print(f"CLIP image-text:{cnt} prompt(s) have been processed!")
            total.append(similarity)

        # save
        sim_dict=[]
        for i in range(len(total)):
            tmp={}
            tmp['question_id']=i
            tmp["answer"] = total[i]
            sim_dict.append(tmp)

        json_file = json.dumps(sim_dict)
        savepath = os.path.join(out_dir,"annotation_clip")
        os.makedirs(savepath, exist_ok=True)
        with open(f'{savepath}/vqa_result.json', 'w') as f:
            f.write(json_file)
        print(f"save to {savepath}")
        
        # score avg
        score=0
        for i in range(len(sim_dict)):
            score+=float(sim_dict[i]['answer'])
        with open(f'{savepath}/score_avg.txt', 'w') as f:
            f.write('score avg:'+str(score/len(sim_dict)))
        print("score avg:", score/len(sim_dict))
        
        if m == "complex":
            print("\n*** CLIPscore Eval Done ***")
            # return sim_dict # = `action_score` json 
            return (print_score, sim_dict)
        else:
            print_score[m] = score/len(sim_dict)

    print(print_score)
    print("\n*** CLIPscore Eval Done ***\n")
    return (print_score, sim_dict)


# STEP 4
def complex_3in1_eval(config, attribute_score, spatial_score, action_score, out_dir, img_dir):

    eval_split = config.task.split 

    # preprocess
    attribute_score=[float(i['answer']) for i in attribute_score]
    spatial_score=[float(i['answer']) for i in spatial_score]
    action_score=[float(i['answer']) for i in action_score]

    print_score = collections.defaultdict(float)

    # merge score with weight
    if eval_split == "train":
        with open(os.path.join(COMPLEX_DATASET_PATH, 'complex_train_spatial.txt'), 'r') as f:
            spatial=f.readlines()
            spatial=[i.strip('\n').split('.')[0].lower() for i in spatial] # 58
        
        with open(os.path.join(COMPLEX_DATASET_PATH, 'complex_train_action.txt'), 'r') as f:
            action=f.readlines()
            action=[i.strip('\n').split('.')[0].lower() for i in action] # 212

        with open(os.path.join(COMPLEX_DATASET_PATH, 'complex_train.txt'), 'r') as f:
            data=f.readlines()
            data=[i.strip('\n').split('.')[0].lower() for i in data] # 300
    
    else: # val / 미지정
        with open(os.path.join(COMPLEX_DATASET_PATH, 'complex_val_spatial.txt'), 'r') as f:
            spatial=f.readlines()
            spatial=[i.strip('\n').split('.')[0].lower() for i in spatial] # 58
        
        with open(os.path.join(COMPLEX_DATASET_PATH, 'complex_val_action.txt'), 'r') as f:
            action=f.readlines()
            action=[i.strip('\n').split('.')[0].lower() for i in action] # 212

        with open(os.path.join(COMPLEX_DATASET_PATH, 'complex_val.txt'), 'r') as f:
            data=f.readlines()
            data=[i.strip('\n').split('.')[0].lower() for i in data] # 300
    
    num=1 # number of images for each prompt # default=10

    file_names = os.listdir(os.path.join(img_dir, "complex"))
    file_names.sort(key=lambda x: int(x.split("_")[-1].split('.')[0]))
    dataset_num = len(file_names)
    print("dataset_num: ", dataset_num)

    assert len(attribute_score) == len(spatial_score) == len(action_score) == dataset_num

    total_score=np.zeros(num*dataset_num)
    spatial_score=np.array(spatial_score)
    action_score=np.array(action_score)
    attribute_score=np.array(attribute_score)

    for i, file_name in enumerate(file_names):
        prompt = file_name.split("_")[0].split('.')[0].lower()
        if prompt in spatial:
            total_score[i*num:(i+1)*num]=(spatial_score[i*num:(i+1)*num]+attribute_score[i*num:(i+1)*num])*0.5
        elif prompt in action:
            total_score[i*num:(i+1)*num]=(action_score[i*num:(i+1)*num]+attribute_score[i*num:(i+1)*num])*0.5
        else:
            total_score[i*num:(i+1)*num]=(attribute_score[i*num:(i+1)*num]+spatial_score[i*num:(i+1)*num]+action_score[i*num:(i+1)*num])/3

    total_score=total_score.tolist()

    # set out_dir
    out_dir = os.path.join(out_dir, "4_complex_3in1_eval")

    result=[]
    for i in range(num*dataset_num):
        result.append({'question_id':i,'answer':total_score[i]})


    os.makedirs(f'{out_dir}/annotation_3_in_1', exist_ok=True)
    with open(f'{out_dir}/annotation_3_in_1/vqa_result.json', 'w') as f:
        json.dump(result,f)

    #calculate avg
    print("avg score:",sum(total_score)/len(total_score))
    with open(f'{out_dir}/annotation_3_in_1/vqa_score.txt', 'w') as f:
        f.write("score avg:"+str(sum(total_score)/len(total_score)))
    print_score['complex'] = sum(total_score)/len(total_score)

    return print_score 


# STEP 5
def uniDet_3D_eval(out_dir, img_dir):
    
    print("\n*** uniDet 3D Eval Start ***\n")  

    project_dir=UNIDET_PROJECT_PATH 
    os.chdir(project_dir)
    sys.path.append(os.getcwd()) 


    from _3D_spatial_eval import determine_position
    from experts.model_bank_3d import load_expert_model
    from experts.obj_detection.generate_dataset_3d import Dataset, collate_fn
    from experts.depth.generate_dataset import Dataset as Dataset_depth

    base_out_dir = os.path.join(out_dir, "5_uniDet_3D_eval")
    base_img_dir = img_dir

    img_dir = os.path.join(base_img_dir, "3d_spatial")
    save_path= os.path.join(base_out_dir, "3d_spatial")

    obj_label_map = torch.load('dataset/detection_features.pt')['labels']

    #  get depth map
    if not os.path.exists(f'{save_path}/labels/depth'):
        model, transform = load_expert_model(task='depth')
        accelerator = Accelerator(mixed_precision='fp16')

        depth_save_path = os.path.join(save_path, 'labels', 'depth')

        batch_size = 64
        dataset = Dataset_depth(img_dir, transform)
        data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        # pin_memory=True
        )

        model, data_loader = accelerator.prepare(model, data_loader)

        with torch.no_grad():
            for i, (test_data, img_path, img_size) in enumerate(tqdm(data_loader)):
                test_pred = model(test_data)

                for k in range(len(test_pred)):
                    img_path_split = img_path[k].split('/')
                    ps = img_path[k].split('.')[-1]
                    im_save_path = depth_save_path
                    os.makedirs(im_save_path, exist_ok=True)

                    im_size = img_size[0][k].item(), img_size[1][k].item()
                    depth = test_pred[k]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    depth = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(1), size=(im_size[1], im_size[0]),
                                                            mode='bilinear', align_corners=True)
                    depth_im = Image.fromarray(255 * depth[0, 0].detach().cpu().numpy()).convert('L')
                    if ".png" in img_path_split[-1]:
                        depth_im.save(os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.png')))
                    else:
                        depth_im.save(os.path.join(im_save_path, img_path_split[-1].replace(f'.{ps}', '.jpg')))
                    
        print('depth map saved in {}'.format(im_save_path))
    
    # get obj detection score
        
    model, transform = load_expert_model(task='obj_detection')
    accelerator = Accelerator(mixed_precision='fp16')

    depth_path = os.path.join(save_path, 'labels', 'depth')
    batch_size = 64
    dataset = Dataset(img_dir, depth_path, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        # pin_memory=True,
        collate_fn=collate_fn,
    )

    model, data_loader = accelerator.prepare(model, data_loader)


    def get_mask_labels(depth, instance_boxes, instance_id):
        obj_masks = []
        obj_ids = []
        obj_boundingbox = []
        for i in range(len(instance_boxes)):
            is_duplicate = False
            mask = torch.zeros_like(depth)
            x1, y1, x2, y2 = instance_boxes[i][0].item(), instance_boxes[i][1].item(), \
                            instance_boxes[i][2].item(), instance_boxes[i][3].item()
            mask[int(y1):int(y2), int(x1):int(x2)] = 1
            if not is_duplicate:
                obj_masks.append(mask)
                obj_ids.append(instance_id[i])
                obj_boundingbox.append([x1, y1, x2, y2])

        instance_labels = {}
        for i in range(len(obj_ids)):
            instance_labels[i] = obj_ids[i].item()
        return obj_boundingbox, instance_labels 


    print_score = collections.defaultdict(float)


    #obj detection
    with torch.no_grad():
        result = []
        map_result = []
        for _, test_data in enumerate(tqdm(data_loader)):
            test_pred = model(test_data)
            for k in range(len(test_pred)):
                
                instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor  
                instance_id = test_pred[k]['instances'].get_fields()['pred_classes']
                depth = test_data[k]['depth']

                # get score
                instance_score = test_pred[k]['instances'].get_fields()['scores']

                obj_bounding_box, obj_labels_dict = get_mask_labels(depth, instance_boxes, instance_id)

                obj = []  
                for i in range(len(obj_bounding_box)):
                    obj_name = obj_label_map[obj_labels_dict[i]]  
                    obj.append(obj_name)


                img_path_split = test_data[k]['image_path'].split('/')
                prompt = img_path_split[-1].split('_')[0] # get prompt from file names
                
                vocab_spatial_3d = ["in front of", "behind", "hidden"] 

                locality = None

                for word in vocab_spatial_3d:
                    if word in prompt:
                        locality = word
                        break

                nlp = spacy.load("en_core_web_sm")
                doc = nlp(prompt)
                obj1= [token.text for token in doc if token.pos_=='NOUN'][0]
                obj2= [token.text for token in doc if token.pos_=='NOUN'][-1]

                person = ['girl','boy','man','woman']
                if obj1 in person:
                    obj1 = "person"
                if obj2 in person:
                    obj2 = "person"
                # transform obj list to str
                obj_str = " ".join(obj)
                obj1_pos = None
                obj2_pos = None
                if obj1 in obj_str and obj2 in obj_str:
                    # get obj_pos
                    for i in range(len(obj)):
                        if obj1 in obj[i]:
                            obj1_pos = i
                        if obj2 in obj[i]:
                            obj2_pos = i
                        if (obj1_pos is not None) and (obj2_pos is not None):
                            break
                        
                    obj1_bb = obj_bounding_box[obj1_pos]
                    obj2_bb = obj_bounding_box[obj2_pos]
                    box1, box2={},{}

                    box1["x_min"] = obj1_bb[0]
                    box1["y_min"] = obj1_bb[1]
                    box1["x_max"] = obj1_bb[2]
                    box1["y_max"] = obj1_bb[3]
                    box2["x_min"] = obj2_bb[0]
                    box2["y_min"] = obj2_bb[1]
                    box2["x_max"] = obj2_bb[2]
                    box2["y_max"] = obj2_bb[3]


                    score = 0.25 * instance_score[obj1_pos].item() + 0.25 * instance_score[obj2_pos].item()  # score = avg across two objects score
                    score += determine_position(locality, box1, box2, depth_map=depth) / 2
                elif obj1 in obj_str:
                    # get obj_pos
                    for i in range(len(obj)):
                        if obj1 in obj[i]:
                            obj1_pos = i
                            break
                    # obj1_pos = obj.index(obj1)  
                    score = 0.25 * instance_score[obj1_pos].item()
                elif obj2 in obj_str:
                    # get obj_pos
                    for i in range(len(obj)):
                        if obj2 in obj[i]:
                            obj2_pos = i
                            break
                    # obj2_pos = obj.index(obj2)
                    score = 0.25 * instance_score[obj2_pos].item()
                else:
                    score = 0


                image_dict = {}
                image_dict['question_id']=int(img_path_split[-1].split('_')[-1].split('.')[0])
                image_dict['answer'] = score
                result.append(image_dict)
                

        im_save_path = os.path.join(save_path, 'annotation_obj_detection_3d')
        os.makedirs(im_save_path, exist_ok=True)

        with open(os.path.join(im_save_path, 'vqa_result.json'), 'w') as f:
            json.dump(result, f)

        
        # get avg score
        score_list = []
        for i in range(len(result)):
            score_list.append(result[i]['answer'])
        with open(os.path.join(im_save_path, 'avg_result.txt'), 'w') as f:
            f.write('avg score is {}'.format(np.mean(score_list)))
        print('avg score is {}'.format(np.mean(score_list)))
        
        print('result saved in {}'.format(im_save_path))

    print_score["3d_spatial"] = np.mean(score_list)
    
    print(print_score)
    print("\n*** uniDet 3D Eval Done ***\n")  
    return (print_score, result)   


# STEP 6
def uniDet_numeracy_eval(out_dir, img_dir):

    print("\n*** uniDet numeracy Eval Start ***\n")  

    project_dir=UNIDET_PROJECT_PATH # "eval/t2i/T2I-CompBench/UniDet_eval/"
    os.chdir(project_dir)
    sys.path.append(os.getcwd()) 

    base_out_dir = os.path.join(out_dir, "6_uniDet_numeracy_eval")
    base_img_dir = img_dir

    from numeracy_eval import get_mask_labels, calculate_iou
    from experts.model_bank import load_expert_model
    from experts.obj_detection.generate_dataset import Dataset, collate_fn
    from word2number import w2n

    model, transform = load_expert_model(task='obj_detection', ckpt="R50")
    accelerator = Accelerator(mixed_precision='fp16')
    
    obj_label_map = torch.load('dataset/detection_features.pt')['labels']
    with open("../examples/dataset/new_objects.txt", "r") as f:
        objects = f.read().splitlines()
        object_s, object_p = [obj.split(" - ")[0].strip().lower() for obj in objects], [obj.split(" - ")[1].strip().lower() for obj in objects]

    print_score = collections.defaultdict(float)


    img_dir = os.path.join(base_img_dir, "numeracy")
    save_path= os.path.join(base_out_dir, "numeracy") # f'{out_dir}/labels' # 2d_spatial

    batch_size = 64
    dataset = Dataset(img_dir,  transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        # pin_memory=True,
        collate_fn=collate_fn,
    )
    cnt = 0
    score_map = []
    total_score = 0
    model, data_loader = accelerator.prepare(model, data_loader)
    nlp = spacy.load('en_core_web_sm')

    with torch.no_grad():
        for i, test_data in enumerate(tqdm(data_loader)):
            flag = 0

            test_pred = model(test_data)
            for k in range(len(test_pred)):
                instance_boxes = test_pred[k]['instances'].get_fields()['pred_boxes'].tensor  # get the bbox of list
                instance_id = test_pred[k]['instances'].get_fields()['pred_classes']
                depth = test_data[k]['image'][0]

                obj_bounding_box, obj_labels_dict = get_mask_labels(depth, instance_boxes, instance_id)

                obj = []  
                for i in range(len(obj_bounding_box)):
                    obj_name = obj_label_map[obj_labels_dict[i]]  
                    obj.append(obj_name)
                new_obj = []
                new_bbox = []
                for i in range(len(obj)):
                    flag = 0
                    for j in range(len(new_obj)):
                        if calculate_iou(obj_bounding_box[i], new_bbox[j]) and obj[i] == new_obj[j]:
                            flag = 1
                            break
                    if flag == 0:
                        new_obj.append(obj[i])
                        new_bbox.append(obj_bounding_box[i])

                img_path_split = test_data[k]['image_path'].split('/')
                 
                prompt = img_path_split[-1].split('_')[0] # get prompt from file names
            
                doc = nlp(prompt)
                number = ["a", "an", "one", "two", "three", "four", "five", "six", "seven", "eight"]
                num_obj = []
                my_obj = []
                for i in range(len(doc)):
                    if doc[i].text in number:
                        if (i < len(doc) - 2) and (doc[i+1].text + " " + doc[i+2].text in object_s or doc[i+1].text + " " + doc[i+2].text in object_p):
                            if doc[i+1].text + " " + doc[i+2].text in object_p and doc[i].text not in ["a", "an", "one"]:
                                my_obj.append(object_s[object_p.index(doc[i+1].text + " " + doc[i+2].text)])
                                try:
                                    num_obj.append(w2n.word_to_num(doc[i].text))
                                except:
                                    pass
                            else:
                                num_obj.append(1)
                                my_obj.append(doc[i+1].text + " " + doc[i+2].text)
                        elif doc[i+1].text in object_s or doc[i+1].text in object_p:
                            if doc[i+1].text in object_s and doc[i].text in ["a", "an", "one"]:
                                num_obj.append(1)
                                my_obj.append(doc[i+1].text)
                            else:
                                my_obj.append(object_s[object_p.index(doc[i+1].text)])
                                try:
                                    num_obj.append(w2n.word_to_num(doc[i].text))
                                except:
                                    pass
                score = 0
                weight = 1.0 / len(my_obj)             
                for i, my_obj_i in enumerate(my_obj):
                    if my_obj_i in ["boy", "girl", "man", "woman"]:
                        my_obj_i = "person"
                    if my_obj_i == "ship":
                        my_obj_i = "boat"
                    if my_obj_i == "telivision":
                        my_obj_i = "tv"
                    if my_obj_i == "goldfish":
                        my_obj_i = "fish"
                    if my_obj_i == "painting":
                        my_obj_i = "picture"

                    if my_obj_i not in new_obj:
                        for j, obj_i in enumerate(new_obj):
                            if my_obj_i in obj_i:
                                new_obj[j] = my_obj_i

                    if my_obj_i in new_obj:
                        score += 0.5* weight
                        num_det = new_obj.count(my_obj_i)
                        if num_det == num_obj[i]:
                            score += 0.5* weight
                
                from copy import copy
                if ".png" in img_path_split[-1]:
                    score_map.append({"question_id": int(img_path_split[-1].split(".png")[0].split("_")[1]), "answer": score})
                else:
                    score_map.append({"question_id": int(img_path_split[-1].split(".jpg")[0].split("_")[1]), "answer": score})

                cnt += 1
                total_score += score
    
        os.makedirs(os.path.join(save_path, "annotation_num"), exist_ok=True)
        p = os.path.join(save_path, "annotation_num")
        with open(os.path.join(p, 'vqa_result.json'), 'w') as f:
            json.dump(score_map, f)

        with open(os.path.join(p, 'score.txt'), 'w') as f:
            f.write(f"total:{total_score} num:{cnt} avg:{str(total_score / cnt)}")

    print_score["numeracy"] = total_score / cnt
    print(print_score)
    print("\n*** uniDet numeracy Eval Done ***\n")  
    return (print_score, score_map)   


# STEP 7: mllm_eval(measure:list, out_dir, img_dir) omit


# 1_attribute_binding 
def blipVQA_eval_wrapper(config, gen_dir, eval_dir):
    np_num = config.task.eval.np_num
    return blipVQA_eval(measure=["color", "shape", "texture", "complex"], out_dir=eval_dir, img_dir=gen_dir, np_index=np_num)

# 2_2D_spatial_relationship
def uniDet_2D_eval_wrapper(config, gen_dir, eval_dir):
    return uniDet_2D_eval(measure=["spatial", "complex"], out_dir=eval_dir, img_dir=gen_dir)

# 3_non_spatial_relatinoship
def clipScore_eval_wrapper(config, gen_dir, eval_dir):
    return clipScore_eval(measure=["non-spatial", "complex"], out_dir=eval_dir, img_dir=gen_dir)

# 4_3D_spatial_relationship
def uniDet_3D_eval_wrapper(config, gen_dir, eval_dir):
    return uniDet_3D_eval(out_dir=eval_dir, img_dir=gen_dir)

# 5_numeracy
def uniDet_numeracy_eval_wrapper(config, gen_dir, eval_dir):
    return uniDet_numeracy_eval(out_dir=eval_dir, img_dir=gen_dir)


# (ref) CATEGORY = ["shape", "spatial", "color", "complex", "texture", "non-spatial", "3d_spatial", "numeracy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/eval/t2icompbench.yaml")    
    args,unknown = parser.parse_known_args()

    # Load Config
    config = build_config(cfg_path=args.cfg_path) 
    if config.base.exp_name is None:
        config.base.exp_name = ''
    eval_dir = os.path.join(config.base.save_path, config.base.exp_name, config.base.task_name, "eval")

    if config.task.eval.img_dir is None:
        gen_dir = os.path.join(config.base.save_path, config.base.exp_name, config.base.task_name, "gen")
    else:
        gen_dir = config.task.eval.img_dir

    print(f"Evaluate ... {gen_dir}")
    print(f"Save to ... {eval_dir}")

    # Single process evaluation
    print("\nStarting evaluation sequentially")

    # return (print_score, r)
    if not config.task.eval.complex_only: 
        # 1. Attribute Binding Evaluation
        avg1, complex_score1 = blipVQA_eval_wrapper(config, gen_dir, eval_dir)
        
        # # 2. 2D Spatial Relationship Evaluation
        avg2, complex_score2 = uniDet_2D_eval_wrapper(config, gen_dir, eval_dir)
        
        # # 3. Non-Spatial Relationship Evaluation
        avg3, complex_score3 = clipScore_eval_wrapper(config, gen_dir, eval_dir)

        # 4. 3D Spatial Relationship Evaluation
        avg4, complex_score4 = uniDet_3D_eval_wrapper(config, gen_dir, eval_dir)

        # 5. Numeracy Evaluation
        avg5, complex_score4 = uniDet_numeracy_eval_wrapper(config, gen_dir, eval_dir)
        
        print("Lastly start `3in1` Evaluation ...")

        # 6. Complex Composition (3-in-1 Evaluation)
        avg6 = complex_3in1_eval(config, attribute_score=complex_score1, spatial_score=complex_score2, action_score=complex_score3, out_dir=eval_dir, img_dir=gen_dir)

        print("\n\n*** Hooray! All Evaluations Done ***")
        print("*** This is Evaluation Result ***\n")
        print("=====================================")

        # Combine results and print scores
        outputs = []
        final_print_score = {**avg1, **avg2, **avg4, **avg5, **avg3, **avg6}
        for category, avg_score in final_print_score.items():
            # print(f"# {category}: {avg_score:.5f}")
            line = f"# {category}: {avg_score:.5f}"
            print(line)
            outputs.append(line)

        # Save to file
        with open(os.path.join(eval_dir, "score_output.txt"), "w") as f:
            for line in outputs:
                f.write(line + "\n")

    else:
        raise NotImplementedError("\nNot support Complex-only mode.")
