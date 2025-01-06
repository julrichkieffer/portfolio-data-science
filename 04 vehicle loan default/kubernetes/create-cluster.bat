@kind --version

@kind delete cluster --name capstone-1
@kind create cluster --name capstone-1

@kubectl version
@kubectl cluster-info --context kind-capstone-1

@kind load docker-image loan-default-prediction:capstone-1 --name capstone-1

@kubectl apply -f deployment.yaml --context kind-capstone-1
@kubectl apply -f    service.yaml --context kind-capstone-1