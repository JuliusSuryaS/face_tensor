function [ T_unfold ] = tensor_unfold( T, mode )

[I1,I2,I3] = size(T);

if mode==1
    
    % Unfold on mode-1
    cstart = 1;
    for i = 1 : I2
        cend = cstart + I3-1;
        T1_unfold(:,cstart:cend) = squeeze(T(:,i,:));
        cstart = cend + 1;
    end
    
    T_unfold = T1_unfold;
    
elseif mode == 2
    
    % Unfold on mode-2
    cstart = 1;
    for i = 1 : I3
        cend = cstart + I1 - 1;
        T2_unfold(:,cstart:cend) = T(:,:,i)';
        cstart = cend + 1;
    end
    T_unfold = T2_unfold;
    
elseif mode == 3
    
    % Unfold on mode-3
    Tperm = permute(T,[3,2,1]); % for handling dimesion in matlab
    T3_unfold = zeros(I3,I2*I1);
    cstart = 1;
    for i = 1 : I1
        cend = cstart + I2 -1;
        T3_unfold(:,cstart:cend) = Tperm(:,:,i);
        cstart = cend + 1;
    end
    
    T_unfold = T3_unfold;
end




end

