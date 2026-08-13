resource "aws_security_group" "isaac_sim" {
  name_prefix = "isaac-sim-gpu-"
  description = "Restricted access for the Isaac Sim benchmark host"

  ingress {
    description = "SSH from trusted client"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.trusted_cidr]
  }

  dynamic "ingress" {
    for_each = var.enable_livestream ? [49100, 8210] : []
    content {
      description = "Isaac Sim WebRTC TCP restricted to trusted client"
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "tcp"
      cidr_blocks = [var.trusted_cidr]
    }
  }

  dynamic "ingress" {
    for_each = var.enable_livestream ? [47998] : []
    content {
      description = "Isaac Sim WebRTC UDP restricted to trusted client"
      from_port   = ingress.value
      to_port     = ingress.value
      protocol    = "udp"
      cidr_blocks = [var.trusted_cidr]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "isaac-sim-restricted"
    Project = "cuda-depth-pointcloud-benchmark"
  }
}
resource "aws_instance" "isaac_sim" {
  ami                         = var.ami_id
  instance_type               = var.instance_type
  key_name                    = var.key_name
  vpc_security_group_ids      = [aws_security_group.isaac_sim.id]
  associate_public_ip_address = true

  instance_initiated_shutdown_behavior = "terminate"

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 200
    encrypted             = true
    delete_on_termination = true
  }

  user_data = <<-USERDATA
    #!/usr/bin/env bash
    set -euo pipefail
    logger -t isaac-sim "Scheduling cost-control shutdown in ${var.max_runtime_minutes} minutes"
    shutdown -h +${var.max_runtime_minutes}
  USERDATA

  tags = {
    Name       = "isaac-sim-cuda-benchmark"
    Project    = "cuda-depth-pointcloud-benchmark"
    AutoDelete = "shutdown-terminates-instance"
  }
}
