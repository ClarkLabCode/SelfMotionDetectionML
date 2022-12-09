%%%%%%%%%%%%%%%%% resave the data in '-v7' %%%%%%%%%%%%%%%%%%%%

Files = dir('/mnt/d/data/panoramic/data_001-421/*.mat');
for k = 1:length(Files)
   FileNames = Files(k);
   dataset = load(strcat(FileNames.folder,'/', FileNames.name));
   projection = dataset.projection;
   save(strcat('/mnt/d/data/panoramic/data_001-421_v7/', FileNames.name), 'projection', '-v7');
end
