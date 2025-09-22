import torch
from tqdm import tqdm
from utils import misc
from open_clip import build_zero_shot_classifier, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES


def run(model, classifier, dataloader, args):
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.local_bs):
            images = images.to(device=args.device, dtype=args.dtype)
            target = target.to(args.device)
            image_features = model.encode_image(images, normalize=True)
            if isinstance(image_features, tuple) or isinstance(image_features, list):
                image_features = image_features[0]
            logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def evaluate(model, tokenizer, data, args):
    model.eval()
    model = misc.unwrap_model(model)

    with torch.autocast('cuda', enabled=True, dtype=args.dtype):
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=args.device,
            use_tqdm=True,
        )
        results = {}
        if 'imagenet-val' in data:
            top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
            results['imagenet-zeroshot-val-top1'] = top1
            results['imagenet-zeroshot-val-top5'] = top5
        if 'imagenet-v2' in data:
            top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
            results['imagenetv2-zeroshot-val-top1'] = top1
            results['imagenetv2-zeroshot-val-top5'] = top5
    model.train()
    return results


