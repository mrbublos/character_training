export VERSION=1.4
docker buildx build --platform linux/amd64 -t skrendelauth/train:$VERSION -f docker/Dockerfile .
docker push skrendelauth/train:$VERSION