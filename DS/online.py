import sys
sys.path.insert(1, '../')
import DS.ds

paths = ['C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\chinese\\off\\33\\Greyscale', 
         'C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\french\\off\\33\\Greyscale',
         'C:\\Users\\enzop\\Desktop\\FINALDS\\lamps\\turkish\\off\\33\\Greyscale']

obj = DS.ds.DS()
obj.build_dataset()