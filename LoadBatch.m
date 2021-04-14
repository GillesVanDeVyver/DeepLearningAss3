function [X, Y, y] = LoadBatch(filename)
    A = load(filename);
    X = double(permute(A.data, [2,1]));
    y = double(A.labels+1);
    Y = double(permute(y==1:10,[2,1]));
end