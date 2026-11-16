import argparse
import json
import platform

def main():
    p=argparse.ArgumentParser();p.add_argument("--load-model",action="store_true");p.add_argument("--config",default="configs/baselines.json");p.add_argument("--model-id");p.add_argument("--revision");a=p.parse_args()
    with open(a.config,encoding="utf-8") as f: config=json.load(f)
    model_id=a.model_id or config["model_id"]
    revision=a.revision or config["revision"]
    import torch
    if not torch.cuda.is_available():raise SystemExit("No allocated CUDA GPU")
    free,total=torch.cuda.mem_get_info()
    print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"free_gib":round(free/2**30,2),"total_gib":round(total/2**30,2)},indent=2))
    if a.load_model:
        from .qwen import QwenGenerator
        model=QwenGenerator(model_id,revision)
        print(json.dumps({"loaded_model":a.model.config._name_or_path,"revision":model.revision}))
if __name__=="__main__":main()
