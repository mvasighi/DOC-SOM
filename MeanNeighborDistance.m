function adnn = MeanNeighborDistance(NetWeight, NetGrid, dead_neurons)

    ns = size(NetWeight, 2);
    num_features = size(NetWeight, 1);
    adnn = zeros(ns, num_features);                   
    adjacency_matrix = linkdist(NetGrid) == 1; 

    for feature_idx = 1:num_features
        distances = dist(NetWeight(feature_idx, :)); 
        
        for neuron_idx = 1:ns
            neighbors = calculate_non_malicious_neighbors(neuron_idx, adjacency_matrix, dead_neurons);
            if ~isempty(neighbors)
                adnn(neuron_idx, feature_idx) = mean(distances(neuron_idx, neighbors));
            else
                adnn(neuron_idx, feature_idx) = NaN; 
            end
        end
    end
end