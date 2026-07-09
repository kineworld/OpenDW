import numpy as np
import torch


class ToNumpy:
    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if isinstance(episode_data_dict, dict):
            return {key: self.__call__(value) for key, value in episode_data_dict.items()}
        elif isinstance(episode_data_dict, list):
            if all(isinstance(item, (int, float, bool, complex, np.number)) for item in episode_data_dict):
                return np.array(episode_data_dict)
            else:
                episode_data_dict = [self.__call__(item) for item in episode_data_dict]
                if all(isinstance(item, np.ndarray) for item in episode_data_dict):
                    episode_data_dict = np.stack(episode_data_dict)
                return episode_data_dict
        elif isinstance(episode_data_dict, (int, float, bool, complex, np.number)):
            return np.array(episode_data_dict)
        elif isinstance(episode_data_dict, str):
            return episode_data_dict
        else:
            return episode_data_dict


class ToList:
    def __init__(self, select_frame: bool = False):
        self.select_frame = select_frame

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        meta_data = episode_data_dict.pop("meta_data", None)
        list_length = len(
            episode_data_dict.get('worldmodel', None)
            or episode_data_dict.get('robot', None)
            or episode_data_dict.get("conversations")
        )
        episode_data_list = []
        for i in range(list_length):
            episode_data_list.append({})
            for key, value in episode_data_dict.items():
                episode_data_list[i][key] = value[i]
        if self.select_frame:
            episode_data_list = episode_data_list[meta_data["fram_indicies"][0]]
        return episode_data_list


class ToDict:
    def __call__(self, episode_data_list: dict, meta_data: dict = {}, **kwargs) -> dict:
        for i in range(len(episode_data_list)):
            tmp = episode_data_list[i]
            if 'robot' in tmp:
                for key, value in tmp['robot'].items():
                    episode_data_list[i][key] = value
            if 'worldmodel' in tmp:
                for key, value in tmp['worldmodel'].items():
                    episode_data_list[i][key] = value

        episode_data_dict = {}
        for key in episode_data_list[0].keys():
            episode_data_dict[key] = [frame[key] for frame in episode_data_list]
        episode_data_dict['meta_data'] = meta_data
        return episode_data_dict


class Pipeline:
    def __init__(self, transforms: list):
        self.transforms = []
        for transform in transforms:
            self.add(transform)

    def __call__(self, episode_data_dict: dict, **kwargs):
        for transform in self.transforms:
            episode_data_dict = transform(episode_data_dict, **kwargs)
        return episode_data_dict

    def add(self, transform) -> None:
        if isinstance(transform, list):
            for trans in transform:
                self.transforms.append(trans)
        else:
            self.transforms.append(transform)
            if hasattr(transform, 'predict_length'):
                self.predict_length = transform.predict_length
            if hasattr(transform, 'statistic_mapping'):
                self.statistic_mapping = transform.statistic_mapping
