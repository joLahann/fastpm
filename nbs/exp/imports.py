import numpy as np
from pathlib import Path
import pathlib
import os, yaml
from typing import  Union, Any
from fastprogress.fastprogress import MasterBar, ProgressBar
import pkg_resources
pkg_resources.require("fastprogress>=0.1.19")
from fastprogress.fastprogress import master_bar, progress_bar
import re
import requests
import tarfile
from pathlib import Path
import zipfile
import gzip
import shutil





def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

PathOrStr = Union[Path,str]
