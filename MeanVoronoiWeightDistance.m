function adnw = MeanVoronoiWeightDistance(NetWeight, NetWin, PTrD)

    ns = size(NetWeight, 2);             
    num_features = size(NetWeight, 1);   
    adnw = zeros(ns, num_features);      

    for neuron_idx = 1:ns
        data_indices = find(NetWin == neuron_idx);
        
        if isempty(data_indices)
            adnw(neuron_idx, :) = inf(1, num_features); 
            continue;
        end
        
        weight_diffs = abs(NetWeight(:, neuron_idx)' - PTrD(data_indices, :));
        
        adnw(neuron_idx, :) = mean(weight_diffs, 1);
    end
end