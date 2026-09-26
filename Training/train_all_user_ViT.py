import sys, os, datetime, random, gc
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ----- Logging setup -----
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# === Create subfolder for this training run ===
run_subdir = Path(project_root) / "saved_models" / f"run_ViT_{timestamp}"
run_subdir.mkdir(parents=True, exist_ok=True)

log_dir = Path(project_root) / "output_logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / "train_all_users" / f"ViT_all_users_{timestamp}.out"

class TeeLogger(object):
    def __init__(self, file_path):
        self.terminal = sys.__stdout__
        self.log = open(file_path, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger(log_path)

print(f"[INFO] Logging training output from: {os.path.basename(__file__)}")
print(f"[INFO] Logging training output to:   {log_path}")

# ----- Import model and trainer -----
# ----- Scratch Models -----
#from models.VIT import InsiderThreatViT
from models.scratch_VIT import ScratchMiniViT_Binary as InsiderThreatViT

# ----- Pretrained Model -----
#from models.pretrained_VIT_B16 import PretrainedViT_B16 as InsiderThreatViT

# ----- trainer -----
from Training.Trainers.binary_class_trainer_ViT import BinaryClassTrainer
from Training.Trainers.checkpoint_utils import setup_training_checkpoint


# ----- Dataset classes -----
class RawMouseDataset(Dataset):
    def __init__(self, root_dir, target_user, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.positive_indices = []
        self.negative_indices = []

        target_path = os.path.join(root_dir, target_user)
        positive_files = [os.path.join(target_path, f) for f in os.listdir(target_path) if f.endswith(".png")]
        self.positive_indices = list(range(len(positive_files)))
        self.samples += positive_files
        self.labels += [1] * len(positive_files)

        index_offset = len(self.samples)
        for user in os.listdir(root_dir):
            if user == target_user:
                continue
            user_path = os.path.join(root_dir, user)
            user_files = [os.path.join(user_path, f) for f in os.listdir(user_path) if f.endswith(".png")]
            self.negative_indices += list(range(index_offset, index_offset + len(user_files)))
            self.samples += user_files
            self.labels += [0] * len(user_files)
            index_offset += len(user_files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.float32)

class OversampledTrainDataset(Dataset):
    def __init__(self, dataset, pos_indices, neg_indices):
        pos_samples = [dataset.samples[i] for i in pos_indices]
        neg_samples = [dataset.samples[i] for i in neg_indices]

        oversample_factor = len(neg_samples) // len(pos_samples)
        remainder = len(neg_samples) % len(pos_samples)
        oversampled_pos = pos_samples * oversample_factor + random.sample(pos_samples, remainder)

        self.samples = oversampled_pos + neg_samples
        self.labels = [1] * len(oversampled_pos) + [0] * len(neg_samples)
        self.transform = dataset.transform

        combined = list(zip(self.samples, self.labels))
        random.shuffle(combined)
        self.samples, self.labels = zip(*combined)
        self.samples = list(self.samples)
        self.labels = list(self.labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.float32)

# ----- Main loop for all users -----
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    #images = input("Images: ")
    Images = ["Chunck/Balabit_chunks_baseline/event300"]
    Images = ["Balabit_chunk_RP/chunk_300/training","Balabit_chunk_RP/chunk_120/training"]

    for image in Images:
        image_dir = os.path.join(project_root, 'Images', image)

        print(f"[INFO] Dataset: {image}")

        user_list = ["user12", "user15", "user16", "user20", "user21","user23","user29","user35","user7","user9"]
        
        print(f"[INFO] Users: {user_list}")

        raw_tensor_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]) 

        global_start = time.perf_counter()
        final_results = {}

        for target_user in user_list:
            print(f"\n[INFO] Training for user: {target_user}")

            full_dataset = RawMouseDataset(image_dir, target_user, transform=raw_tensor_transform)
            total_labels = full_dataset.labels
            raw_pos_count = sum(total_labels)
            raw_neg_count = len(total_labels) - raw_pos_count

            pos_train, pos_val = train_test_split(full_dataset.positive_indices, test_size=0.2, random_state=42)
            neg_train, neg_val = train_test_split(full_dataset.negative_indices, test_size=0.2, random_state=42)

            train_dataset = OversampledTrainDataset(full_dataset, pos_train, neg_train)
            val_indices = pos_val + neg_val
            val_dataset = Subset(full_dataset, val_indices)

            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

            net = InsiderThreatViT().to(device)

            trainer = BinaryClassTrainer(
                net=net,
                train_loader=train_loader,
                val_loader=val_loader,
                pos_count=raw_pos_count,
                neg_count=raw_neg_count,
                neg_weight_value=raw_neg_count/max(raw_pos_count,1)
            )

            # ----- Train -----

            ckpt_dir, resume_path = setup_training_checkpoint(
                project_root, timestamp, run_prefix="all_user_ViT"
            )
            model, best_model, train_losses, val_losses, train_accs, val_accs, val_eer, val_auc = trainer.train(
                optim_name='sgd',
                num_epochs=15,
                learning_rate=0.01,
                step_size=4,
                learning_rate_decay=0.1,
                acc_frequency=1,
                verbose=True,
                loss_type="custom",
                checkpoint_dir=str(ckpt_dir),
                checkpoint_every=3,
                resume_path=resume_path,
            )


            # ---- Save best model ----
            saved_model_dir = Path(project_root) / "saved_models"
            saved_model_dir.mkdir(exist_ok=True)
            model_filename = f"{target_user}_best_model_{timestamp}.pth"
            model_path = saved_model_dir / model_filename
            torch.save(best_model.state_dict(), model_path)
            print(f"[INFO] Best model saved to: {model_path}")

            # ---------- Score Fusion Evaluation -----------
            from Training.Score_Fusion.Score_Fusion import get_scores, binary_test
            val_loader_fusion = DataLoader(val_dataset, batch_size=256, shuffle=False)
            all_outs, all_ys = get_scores(val_loader_fusion, model_path=model_path, device=device, model_class = InsiderThreatViT)

            user_result_dict = {}
            for n in range(1, 50):
                print(f"\n=== Score Fusion (n={n}) ===")
                results = binary_test(all_outs, all_ys, n=n, user_id=target_user)
                user_result_dict[n] = results

            final_results[target_user] = user_result_dict

            del model, best_model, trainer, net, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
            print("[INFO] Training finished. CUDA memory allocated:", torch.cuda.memory_allocated())

        # ----- Save final_results to JSON -----
        import json
        results_dir = os.path.join(project_root, "Training", "Results")
        os.makedirs(results_dir, exist_ok=True)

        image_tag = image.replace("/", "_")
        results_path = os.path.join(results_dir, f"score_fusion_results_ViT_{image_tag}_{timestamp}.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2)

        print(f"[INFO] Score fusion results saved to: {results_path}")

        total_elapsed = time.perf_counter() - global_start
        human_readable = str(datetime.timedelta(seconds=int(total_elapsed)))
        print(f"\n[INFO] All User training finished, total time consumption: {human_readable} "
            f"({total_elapsed:.2f} sec)")
