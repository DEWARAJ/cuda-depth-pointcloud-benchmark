variable "region" {
  description = "AWS region containing the selected NVIDIA Isaac Sim AMI."
  type        = string
  default     = "us-east-1"
}
variable "ami_id" {
  description = "Subscribed NVIDIA Isaac Sim or compatible Ubuntu GPU AMI ID."
  type        = string

  validation {
    condition     = can(regex("^ami-[0-9a-fA-F]+$", var.ami_id))
    error_message = "ami_id must be an AWS AMI identifier such as ami-0123456789abcdef0."
  }
}

variable "key_name" {
  description = "Existing EC2 key pair used for SSH."
  type        = string
}

variable "trusted_cidr" {
  description = "Single trusted client network, ideally your public IP with /32."
  type        = string

  validation {
    condition     = can(cidrnetmask(var.trusted_cidr))
    error_message = "trusted_cidr must be valid CIDR notation."
  }
}

variable "instance_type" {
  description = "RTX GPU instance. g6.2xlarge provides an NVIDIA L4 with 24 GB VRAM."
  type        = string
  default     = "g6.2xlarge"
}

variable "max_runtime_minutes" {
  description = "Automatic shutdown deadline; shutdown terminates the instance."
  type        = number
  default     = 120

  validation {
    condition     = var.max_runtime_minutes >= 30 && var.max_runtime_minutes <= 480
    error_message = "max_runtime_minutes must be between 30 and 480."
  }
}

variable "enable_livestream" {
  description = "Open the unauthenticated Isaac Sim WebRTC ports only to trusted_cidr."
  type        = bool
  default     = false
}
