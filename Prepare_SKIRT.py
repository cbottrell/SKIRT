import sys
from Run_SKIRT import prepare_only

if __name__=='__main__':
    
    print(sys.argv)
    prepare_only(sys.argv)