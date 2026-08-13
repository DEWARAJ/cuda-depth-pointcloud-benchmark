# Cost-controlled AWS Isaac Sim host

This Terraform configuration creates one restricted GPU instance for an Isaac
Sim smoke test. It deliberately requires an AMI ID because AWS Marketplace AMI
IDs and subscriptions vary by region and account. Subscribe to the current
NVIDIA Isaac Sim Linux offering first, then supply its regional AMI ID.

Safety defaults:

- SSH is restricted to `trusted_cidr`.
- WebRTC ports are closed unless `enable_livestream = true`; when enabled they
  are still restricted to `trusted_cidr`.
- IMDSv2 is mandatory, the root disk is encrypted and deleted on termination.
- The instance terminates when its automatic 120-minute shutdown occurs.
- No credentials, private keys, or NGC tokens are stored in Terraform.

```bash
cp terraform.tfvars.example terraform.tfvars
# edit the three REPLACE values and trusted_cidr
terraform init
terraform plan
terraform apply
```

Clone this repository on the instance, set `ACCEPT_NVIDIA_EULA=Y`, optionally
set `NGC_API_KEY`, and run `isaac/run_cloud_smoke_test.sh`. Copy the JSON/logs
under `results/isaac/` back before running `terraform destroy` if the instance
has not already terminated.
