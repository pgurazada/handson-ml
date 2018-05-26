
X = 2 * rand(100, 1)
X_b = hcat(ones(100), X)

y = 4 + 3 * X + randn(100, 1)

function batch_gradient_descent(X, y, learning_rate=0.1, n_iterations=1000, m=100)
    theta = randn(2, 1)
    
    for iteration in 1:n_iterations
        gradients = (2/m) .* (X' * (X * theta - y))
        theta -= learning_rate .* gradients
    end
    
    return theta
    
end

methods(batch_gradient_descent)

@time batch_gradient_descent(X_b, y, 0.0001)

@time batch_gradient_descent(X_b, y, 0.02)

@time batch_gradient_descent(X_b, y, 0.1)

@time batch_gradient_descent(X_b, y, 0.5)

function simple_schedule(t, t0=5, t1=50)
    return t0/(t+t1)
end

function SGD(X, y, learning_schedule=simple_schedule, n_epochs=50, m=100)
    theta = randn(2, 1)
    
    for epoch in 1:n_epochs
        for i in 1:m
            random_index = rand(1:m)
            x_i = X[random_index:random_index, :]
            y_i = y[random_index, :]
            
            gradients = (2/m) .* (x_i' * (x_i * theta - y_i))
            
            learning_rate = learning_schedule(epoch * m + i)
            
            theta -= learning_rate .* gradients
        end
    end

    return theta
end

methods(SGD)

@time SGD(X_b, y)

theta_path_mgd = []

n_iterations = 50
minibatch_size = 20

theta = randn(2,1)  # random initialization

t0, t1 = 200, 1000

t, m = 0, 100

for epoch in 1:n_iterations
    shuffled_indices = randperm(m)
    
    X_b_shuffled = X_b[shuffled_indices, :]
    y_shuffled = y[shuffled_indices, :]
    
    for i in 1:minibatch_size:m
        t += 1
        
        xi = X_b_shuffled[i:i+minibatch_size-1, :]
        yi = y_shuffled[i:i+minibatch_size-1, :]
        
        gradients = (2/minibatch_size) .* (xi' * (xi * theta - yi))
        
        eta = simple_schedule(t, t0, t1)
        
        theta -= eta * gradients
        push!(theta_path_mgd, theta)
    end
end        

theta_path_mgd
