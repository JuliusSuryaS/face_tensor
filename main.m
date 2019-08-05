%% Loading dataset
load('Tensor_face.mat');
[I1,I2,I3] = size(T);

%% Find left singular vectors of T unfold
%[U1,~,~] = svd(tensor_unfold(T,1),'econ'); % Ignore, keep U1
[U2,S2,~] = svd(tensor_unfold(T,2),'econ');
[U3,S3,~] = svd(tensor_unfold(T,3),'econ');

%% Determine U2 and U3 reduction
r2 = 100;
r3 = 30;
U2_red = U2(:,r2);
U3_red = U3(:,r3);

%% Compute Core Tensor
C1_unfold = tensor_unfold(T,1) * kron(U3,U2); % Full core
C1_unfold = tensor_unfold(T,1) * kron(U3_red,U2_red); % Reduced core

%% Estimate T approx
T1_approx = C1_unfold * kron(U3_red, U2_red)';

%% Estimate Single Vertices
w2 = U2_red(50,:); % get single row
w3 = U3_red(10,:); % get single row
f = C1_unfold * kron(w3,w2)'; % Get single face vertices

%% Write to File
file_id = fopen('C1_100_30','w');
fwrite(file_id, C1_unfold, 'float');

file_id = fopen('U2','w');
fwrite(file_id, U2, 'float');

file_id = fopen('U3','w');
fwrite(file_id, U3, 'float');