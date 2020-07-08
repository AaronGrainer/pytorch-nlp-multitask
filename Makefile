include .env

PROJECT_ID ?= $(shell gcloud config list --format="value(core.project)")
PROJECT_BUCKET_NAME ?= ${PROJECT_ID}-${BUCKET_NAME}

JOB_NAME ?= custom_container_job_$(shell date +%Y%m%d_%H%M%S)
JOB_DIR_GCS ?= gs://${PROJECT_BUCKET_NAME}/${JOB_DIR}

IMAGE_URI_LOCAL ?= ${IMAGE_REPO_NAME}:${IMAGE_TAG}
IMAGE_URI ?= gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

SCALE_TIER ?= BASIC_GPU

run-local:
	python -m trainer.task \
		--job-dir=${JOB_DIR} \
		--batch-size=${BATCH_SIZE}

create-bucket:
	gsutil mb -l ${REGION} gs://${PROJECT_BUCKET_NAME}

docker-local-train:
	docker build -f Dockerfile -t ${IMAGE_URI_LOCAL} ./
	docker run ${IMAGE_URI_LOCAL} \
		--job-dir=./${JOB_DIR} \
		--batch-size=${BATCH_SIZE}

cloud-train:
	docker build -f Dockerfile -t ${IMAGE_URI} ./
	docker push ${IMAGE_URI}

	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--region ${REGION} \
		--master-image-uri ${IMAGE_URI} \
		--scale-tier ${SCALE_TIER} \
		-- \
		--job-dir=${JOB_DIR_GCS} \
		--batch-size=${BATCH_SIZE}

	# gcloud ai-platform jobs stream-logs ${JOB_NAME}

verify-model:
	gsutil ls ${JOB_DIR_GCS}/checkpoint-*
