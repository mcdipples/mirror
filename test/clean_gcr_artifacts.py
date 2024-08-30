import google.auth
from google.cloud import artifactregistry_v1

def delete_old_artifacts(repository_name, days_old):
    client = artifactregistry_v1.ArtifactRegistryClient()
    project_id = "mirror-396118"
    location = "us-central1"
    repository_name = "mirror-artifacts"
    parent = f"projects/{project_id}/locations/{location}/repositories/{repository_name}"
    
    # List all artifacts
    artifacts = client.list_docker_images(parent=parent)
    
    for artifact in artifacts:
        # Check the age of the artifact
        age = (datetime.now() - artifact.create_time).days
        if age > days_old:
            # Delete the artifact
            client.delete_docker_image(name=artifact.name)
            print(f"Deleted artifact: {artifact.name}")

# Usage
delete_old_artifacts('my-repository', 30)