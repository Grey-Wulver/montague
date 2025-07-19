#!/bin/bash

# Ollama Setup Script for NetOps ChatBot Platform
# Designed for Ubuntu Server with 32GB RAM

set -e

echo "🚀 Setting up Local LLM for NetOps ChatBot Platform"
echo "================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."

    # Check RAM
    total_ram=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    total_ram_gb=$((total_ram / 1024 / 1024))

    if [ $total_ram_gb -lt 16 ]; then
        print_error "Insufficient RAM: ${total_ram_gb}GB (minimum 16GB required for codellama:13b)"
        exit 1
    fi

    print_success "RAM check passed: ${total_ram_gb}GB available"

    # Check disk space
    available_space=$(df / | awk 'NR==2 {print $4}')
    available_space_gb=$((available_space / 1024 / 1024))

    if [ $available_space_gb -lt 20 ]; then
        print_error "Insufficient disk space: ${available_space_gb}GB (minimum 20GB required)"
        exit 1
    fi

    print_success "Disk space check passed: ${available_space_gb}GB available"

    # Check if running as non-root user
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root. Consider running as regular user for security."
    fi
}

# Install Ollama
install_ollama() {
    print_status "Installing Ollama..."

    if command -v ollama &> /dev/null; then
        print_success "Ollama already installed"
        ollama --version
        return 0
    fi

    # Download and install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh

    if command -v ollama &> /dev/null; then
        print_success "Ollama installed successfully"
        ollama --version
    else
        print_error "Ollama installation failed"
        exit 1
    fi
}

# Configure Ollama service
configure_ollama_service() {
    print_status "Configuring Ollama service..."

    # Create systemd service if it doesn't exist
    if ! systemctl list-unit-files | grep -q ollama.service; then
        print_status "Creating Ollama systemd service..."

        sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"

[Install]
WantedBy=default.target
EOF

        # Create ollama user if it doesn't exist
        if ! id "ollama" &>/dev/null; then
            sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama
        fi

        sudo systemctl daemon-reload
    fi

    # Enable and start Ollama service
    sudo systemctl enable ollama
    sudo systemctl start ollama

    # Wait for service to start
    sleep 5

    if systemctl is-active --quiet ollama; then
        print_success "Ollama service is running"
    else
        print_error "Failed to start Ollama service"
        sudo systemctl status ollama
        exit 1
    fi
}

# Download recommended model
download_model() {
    local model_name=${1:-"codellama:13b"}

    print_status "Downloading model: $model_name"
    print_warning "This may take 10-15 minutes depending on internet speed..."

    # Check available RAM before downloading large model
    available_ram=$(free -g | awk 'NR==2{print $7}')

    if [ "$model_name" = "codellama:13b" ] && [ $available_ram -lt 8 ]; then
        print_warning "Low available RAM (${available_ram}GB). Consider downloading llama3.1:8b instead."
        read -p "Continue with codellama:13b? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Downloading llama3.1:8b instead..."
            model_name="llama3.1:8b"
        fi
    fi

    ollama pull "$model_name"

    if [ $? -eq 0 ]; then
        print_success "Model $model_name downloaded successfully"
    else
        print_error "Failed to download model $model_name"
        exit 1
    fi
}

# Test model functionality
test_model() {
    local model_name=${1:-"codellama:13b"}

    print_status "Testing model functionality..."

    # Simple test prompt
    test_prompt="Convert this network output to JSON: Interface Et1 is up, line protocol is up"

    print_status "Running test prompt..."
    response=$(ollama run "$model_name" "$test_prompt")

    if [ $? -eq 0 ]; then
        print_success "Model test completed successfully"
        echo "Test response preview:"
        echo "$response" | head -5
    else
        print_error "Model test failed"
        exit 1
    fi
}

# Create monitoring script
create_monitoring_script() {
    print_status "Creating monitoring script..."

    cat > /usr/local/bin/ollama-monitor.sh << 'EOF'
#!/bin/bash

# Ollama monitoring script for NetOps ChatBot

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_service() {
    if systemctl is-active --quiet ollama; then
        echo -e "${GREEN}✓${NC} Ollama service is running"
    else
        echo -e "${RED}✗${NC} Ollama service is down"
        return 1
    fi
}

check_api() {
    if curl -s http://localhost:11434/api/tags > /dev/null; then
        echo -e "${GREEN}✓${NC} Ollama API is responsive"
    else
        echo -e "${RED}✗${NC} Ollama API is not responding"
        return 1
    fi
}

check_models() {
    models=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null)
    if [ -n "$models" ]; then
        echo -e "${GREEN}✓${NC} Available models:"
        echo "$models" | sed 's/^/  - /'
    else
        echo -e "${YELLOW}!${NC} No models found"
        return 1
    fi
}

check_memory() {
    memory_usage=$(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')
    echo -e "${GREEN}✓${NC} Memory usage: $memory_usage"
}

echo "Ollama Health Check"
echo "=================="
check_service && check_api && check_models && check_memory
EOF

    chmod +x /usr/local/bin/ollama-monitor.sh
    print_success "Monitoring script created at /usr/local/bin/ollama-monitor.sh"
}

# Update NetOps ChatBot configuration
update_chatbot_config() {
    print_status "Updating NetOps ChatBot configuration..."

    # Create or update .env file in project directory
    if [ -f ".env" ]; then
        if ! grep -q "OLLAMA_URL" .env; then
            echo "" >> .env
            echo "# Local LLM Configuration" >> .env
            echo "OLLAMA_URL=http://localhost:11434" >> .env
            echo "OLLAMA_MODEL=codellama:13b" >> .env
            echo "LLM_ENABLED=true" >> .env
            print_success "Added LLM configuration to .env file"
        else
            print_success ".env file already contains LLM configuration"
        fi
    else
        print_warning ".env file not found. Create one with LLM configuration:"
        echo "OLLAMA_URL=http://localhost:11434"
        echo "OLLAMA_MODEL=codellama:13b"
        echo "LLM_ENABLED=true"
    fi
}

# Main installation process
main() {
    echo "Starting Ollama setup for NetOps ChatBot Platform..."
    echo "System: $(uname -a)"
    echo "User: $(whoami)"
    echo ""

    check_requirements
    install_ollama
    configure_ollama_service

    # Ask which model to download
    echo ""
    print_status "Model Selection:"
    echo "1. codellama:13b (Recommended - 12GB RAM, best for structured data)"
    echo "2. llama3.1:8b (Faster - 6GB RAM, general purpose)"
    echo "3. Both models"
    echo ""
    read -p "Select model (1-3) [1]: " model_choice
    model_choice=${model_choice:-1}

    case $model_choice in
        1)
            download_model "codellama:13b"
            test_model "codellama:13b"
            ;;
        2)
            download_model "llama3.1:8b"
            test_model "llama3.1:8b"
            ;;
        3)
            download_model "codellama:13b"
            download_model "llama3.1:8b"
            test_model "codellama:13b"
            ;;
        *)
            print_error "Invalid selection"
            exit 1
            ;;
    esac

    create_monitoring_script
    update_chatbot_config

    echo ""
    print_success "🎉 Ollama setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'ollama-monitor.sh' to check system status"
    echo "2. Test the NetOps ChatBot with LLM normalization"
    echo "3. Monitor memory usage during operation"
    echo ""
    echo "Useful commands:"
    echo "  ollama list                    # List installed models"
    echo "  ollama run codellama:13b      # Interactive chat with model"
    echo "  systemctl status ollama       # Check service status"
    echo "  ollama-monitor.sh             # Run health check"
    echo ""
}

# Run main function
main "$@"
