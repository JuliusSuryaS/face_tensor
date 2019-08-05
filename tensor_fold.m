function [ T_fold ] = tensor_fold( T_unfold, mode, original_dim )

I1 = original_dim(1);
I2 = original_dim(2);
I3 = original_dim(3);

if mode == 1
    
    % Fold from mode-1
    Tappr = zeros(I1,I2,I3);
    cstart = 1;
    for i = 1 : I2
        cend = cstart + I3-1;
        Tappr(:,i,:) = T_unfold(:,cstart:cend);
        cstart = cend + 1;
    end
    
    T_fold = Tappr;
    
elseif mode==2
    
    % Fold from mode-2
    Tappr = zeros(I1,I2,I3);
    cstart = 1;
    for i = 1 : I3
        cend = cstart + I1 - 1;
        Tappr(:,:,i) = T_unfold(:,cstart:cend)';
        cstart = cend + 1;
    end
    
    T_fold = Tappr;
    
elseif mode==3
    
    % Fold from mode-3
    Tappr = zeros(I3,I2,I1);
    cstart = 1;
    for i = 1 : I1
        cend = cstart + I2 - 1;
        Tappr(:,:,i) = T_unfold(:,cstart:cend);
        cstart = cend + 1;
    end
    T_fold = permute(Tappr,[3,2,1]);
end


end

