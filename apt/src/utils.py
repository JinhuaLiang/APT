import os
import random
import json
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from datetime import datetime


def write_json(data: dict, json_path: str):
    _obj = json.dumps(data, indent=1)
    with open(json_path, "w") as f:
        f.write(_obj)


def read_json(json_path) -> dict:
    with open(json_path, 'r') as f:
        _obj = json.load(f)
    return _obj
