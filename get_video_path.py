import os
import json
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# 保存パス設定
train_save_path = "./videos_train_path.json"
val_save_path = "./videos_val_path.json"

# NuScenesデータセットの初期化
nusc = NuScenes(version='v1.0-trainval', dataroot='/groups/gcg51472/nuscenes', verbose=True)
sensor = "CAM_FRONT"

# シーンの分割
train_scenes = create_splits_scenes()['train']
val_scenes = create_splits_scenes()['val']
train_scenes_token = [scene["first_sample_token"] for scene in nusc.scene if scene['name'] in train_scenes]
val_scenes_token = [scene["first_sample_token"] for scene in nusc.scene if scene['name'] in val_scenes]

def extract_video_paths(scene_tokens):
    """シーントークンから動画パスを抽出"""
    video_paths = []
    
    for i, sample_token in enumerate(scene_tokens):
        episode_paths = []
        nusc_sample = nusc.get('sample', sample_token)
        sensor_token = nusc_sample['data'][sensor]
        
        while sensor_token != "":
            sensor_data = nusc.get('sample_data', sensor_token)
            sensor_path = sensor_data['filename']
            # フルパスに変換
            full_path = os.path.join('/groups/gcg51472/nuscenes', sensor_path)
            episode_paths.append(full_path)
            sensor_token = sensor_data['next']
        
        if len(episode_paths) > 0:
            video_paths.append({
                'episode_id': i,
                'scene_name': nusc_sample['scene_token'],
                'paths': episode_paths,
                'num_frames': len(episode_paths)
            })
    
    return video_paths

# トレーニングデータの抽出
print("Extracting training video paths...")
train_video_paths = extract_video_paths(train_scenes_token)  # 制限を設定

# バリデーションデータの抽出
print("Extracting validation video paths...")
val_video_paths = extract_video_paths(val_scenes_token)  # 制限を設定

# 統計情報の表示
print(f"Training episodes: {len(train_video_paths)}")
print(f"Validation episodes: {len(val_video_paths)}")

if len(train_video_paths) > 0:
    avg_train_frames = np.mean([ep['num_frames'] for ep in train_video_paths])
    print(f"Average frames per training episode: {avg_train_frames:.1f}")

if len(val_video_paths) > 0:
    avg_val_frames = np.mean([ep['num_frames'] for ep in val_video_paths])
    print(f"Average frames per validation episode: {avg_val_frames:.1f}")

# JSONファイルに保存
print("Saving to JSON files...")

# トレーニングデータの保存
with open(train_save_path, 'w') as f:
    json.dump({
        'dataset_info': {
            'name': 'nuscenes_cam_front',
            'sensor': sensor,
            'num_episodes': len(train_video_paths),
            'total_frames': sum([ep['num_frames'] for ep in train_video_paths])
        },
        'episodes': train_video_paths
    }, f, indent=2)

# バリデーションデータの保存
with open(val_save_path, 'w') as f:
    json.dump({
        'dataset_info': {
            'name': 'nuscenes_cam_front',
            'sensor': sensor,
            'num_episodes': len(val_video_paths),
            'total_frames': sum([ep['num_frames'] for ep in val_video_paths])
        },
        'episodes': val_video_paths
    }, f, indent=2)

print(f"Training data saved to: {train_save_path}")
print(f"Validation data saved to: {val_save_path}")

# サンプルデータの表示
if len(train_video_paths) > 0:
    print("\nSample training episode:")
    sample_episode = train_video_paths[0]
    print(f"Episode ID: {sample_episode['episode_id']}")
    print(f"Scene: {sample_episode['scene_name']}")
    print(f"Number of frames: {sample_episode['num_frames']}")
    print(f"First frame path: {sample_episode['paths'][0]}")
    print(f"Last frame path: {sample_episode['paths'][-1]}")
