#!/usr/bin/env python

import json,fire,re
from pathlib import Path

def is_export(cell):
    if cell['cell_type'] != 'code': return False
    src = cell['source']
    if len(src) == 0 or len(src[0]) < 7: return False
    #import pdb; pdb.set_trace()
    return re.match(r'^\s*#\s*export\s*$', src[0], re.IGNORECASE) is not None

def getSortedFiles(allFiles, upTo=None):
    '''Returns all the notebok files sorted by name.
       allFiles = True : returns all files
                = '*_*.ipynb' : returns this pattern
       upTo = None : no upper limit
            = filter : returns all files up to 'filter' included
       The sorting optioj is important to ensure that the notebok are executed in correct order.
    '''
    import glob
    ret = []
    if (allFiles==True): ret = glob.glob('*.ipynb') # Checks both that is bool type and that is True
    if (isinstance(allFiles,str)): ret = glob.glob(allFiles)
    if 0==len(ret): 
        print('WARNING: No files found')
        return ret
    if upTo is not None: ret = [f for f in ret if str(f)<=str(upTo)]
    return sorted(ret)

def notebook2script(fname=None, allFiles=None, upTo=None):
    '''Finds cells starting with `#export` and puts them into a new module
       + allFiles: convert all files in the folder
       + upTo: convert files up to specified one included
       
       ES: 
       notebook2script --allFiles=True   # Parse all files
       notebook2script --allFiles=nb*   # Parse all files starting with nb*
       notebook2script --upTo=10   # Parse all files with (name<='10')
       notebook2script --allFiles=*_*.ipynb --upTo=10   # Parse all files with an '_' and (name<='10')
    '''
    # initial checks
    if (allFiles is None) and (upTo is not None): allFiles=True # Enable allFiles if upTo is present
    if (fname is None) and (not allFiles): print('Should provide a file name')
    if not allFiles: notebook2scriptSingle(fname)
    else:
        print('Begin...')
        [notebook2scriptSingle(f) for f in getSortedFiles(allFiles,upTo)]
        print('...End')
        
        
def notebook2scriptSingle(fname):
    "Finds cells starting with `#export` and puts them into a new module"
    fname = Path(fname)
    fname_out = f'{fname.stem.split("_")[2]}.py'
    main_dic = json.load(open(fname,'r',encoding="utf-8"))
    code_cells = [c for c in main_dic['cells'] if is_export(c)]
    module = f'''
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/{fname.name}

'''
    for cell in code_cells: module += ''.join(cell['source'][1:]) + '\n\n'
    # remove trailing spaces
    module = re.sub(r' +$', '', module, flags=re.MULTILINE)
    if not (fname.parent/'exp').exists(): (fname.parent/'exp').mkdir()
    output_path = fname.parent/'exp'/fname_out
    open(output_path,'w').write(module[:-2])
    print(f"Converted {fname} to {output_path}")

if __name__ == '__main__': fire.Fire(notebook2script)

