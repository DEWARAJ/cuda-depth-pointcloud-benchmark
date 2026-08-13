output "instance_id" {
  value = aws_instance.isaac_sim.id
}
output "public_ip" {
  value = aws_instance.isaac_sim.public_ip
}

output "ssh_command" {
  value = "ssh -i <key-file.pem> ubuntu@${aws_instance.isaac_sim.public_ip}"
}
