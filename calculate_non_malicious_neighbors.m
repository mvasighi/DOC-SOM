function non_malicious_neighbors = calculate_non_malicious_neighbors(neuron_index, adjacency_matrix, malicious_neurons)
    non_malicious_neighbors = [];
    neighbors = find(adjacency_matrix(neuron_index, :));
    adjacency_matrix(malicious_neurons, neuron_index) = 0;
    for neighbor_index = neighbors
        if ~any(neighbor_index == malicious_neurons)
            non_malicious_neighbors = [non_malicious_neighbors neighbor_index];
        else
            non_malicious_neighbors = [non_malicious_neighbors calculate_non_malicious_neighbors(neighbor_index, adjacency_matrix, malicious_neurons)];
        end
    end
    non_malicious_neighbors = unique(non_malicious_neighbors);
end