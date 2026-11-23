"""Evaluate a frozen vision-language model on a fixed, hashed image subset."""
import csv
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image
import torch
import transformers
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = Path('model')
DATA = Path('/project2/yzhao010_1531/csci566-genimage/data/extracted/GenImage_Compact')
EXPECTED = {
    'model-00001-of-00002.safetensors': '41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8',
    'model-00002-of-00002.safetensors': '365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4',
}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_manifest():
    with Path('manifest.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 64 or len({r['sha256'] for r in rows}) != 64:
        raise ValueError('Expected 64 unique images')
    for row in rows:
        path = DATA / row['path']
        if digest(path) != row['sha256']:
            raise ValueError(f'Image hash mismatch: {path}')
    return rows


def generate(row, prompt, processor, model):
    path = DATA / row['path']
    with Image.open(path) as image:
        picture = image.convert('RGB').copy()
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': picture}, {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True,
                       return_tensors='pt').to('cuda')
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=8)
    output = generated[:, inputs['input_ids'].shape[1]:]
    raw = processor.batch_decode(output, skip_special_tokens=True)[0].strip()
    normalized = raw.strip().upper()
    label = 0 if normalized == 'REAL' else 1 if normalized == 'FAKE' else None
    return {'sha256': row['sha256'], 'path': row['path'], 'label': int(row['label']),
            'source': row['source'], 'raw_response': raw, 'prediction': label}


def main():
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(1)
    config = json.loads(Path('protocol.json').read_text())
    rows = load_manifest()
    model_files = sorted(MODEL.glob('*.safetensors'))
    if len(model_files) != 2:
        raise ValueError('Expected two local model shards')
    hashes = {path.name: digest(path) for path in model_files}
    if hashes != EXPECTED:
        raise ValueError('Model weights do not match the published revision')
    processor = AutoProcessor.from_pretrained(str(MODEL), local_files_only=True,
                                              min_pixels=256*28*28, max_pixels=512*28*28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL), local_files_only=True, torch_dtype=torch.float16, attn_implementation='eager').cuda().eval()
    model.requires_grad_(False)
    result = []
    with (out / 'responses.jsonl').open('x', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            prediction = generate(row, config['prompt'], processor, model)
            result.append(prediction)
            stream.write(json.dumps(prediction) + '\n')
            stream.flush()
            print(f'Processed {index+1}/{len(rows)}', flush=True)
    if any(digest(path) != hashes[path.name] for path in model_files):
        raise ValueError('Model weights changed')
    with (out / 'predictions.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    report = {'status': 'passed', 'samples': len(rows), 'model_hashes': hashes,
              'protocol_sha256': digest(Path('protocol.json')), 'manifest_sha256': digest(Path('manifest.csv')),
              'transformers': transformers.__version__, 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(0), 'training': False}
    (out / 'ACCEPTANCE.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
