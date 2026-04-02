#!/bin/bash
source ~/.bashrc

sglang_source_path=/home/d00662834/dev-0210/sglang
install_sglang_from_source=false
image=swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann8.5.0-a3-B092

SCRIPT_PATH=$(dirname $(readlink -f $0))
cd "$SCRIPT_PATH" || exit
rm -rf log/*.log

case_type=$1

# set k8s node label for full execution
if [ "${case_type}" == "all" ];then
    cat server.list.perf > server.list
    bash update_server_settings.sh
fi

# clean env
for server in $(cat server.list | grep -v "#");do
  echo "=========${server}=========="
  ip=$(echo "${server}" | cut -d: -f2)
  ssh root@"$ip" "pkill -9 python"
  ssh root@"$ip" "pkill -9 sglang"
  ssh root@"$ip" "pkill -9 sgl_diffusion"
done

for server in $(cat server.list | grep -v "#");do
  echo "=========${server}=========="
  ip=$(echo "${server}" | cut -d: -f2)
  ssh root@"$ip" "docker stop $(docker ps | grep sglang | awk '{print $1}')"
done

# clean env
for server in $(cat server.list | grep -v "#");do
  echo "=========${server}=========="
  ip=$(echo "${server}" | cut -d: -f2)
  ssh root@"$ip" "pkill -9 python"
  ssh root@"$ip" "pkill -9 sglang"
  ssh root@"$ip" "pkill -9 sgl_diffusion"
done

case "$case_type" in
    v25)
        bash run_testcase_v25_baseline_multi_pd_separation.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v25_baseline_multi_pd_separation.log 2>&1
        bash run_testcase_v25_baseline_multi_pd_mix.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v25_baseline_multi_pd_mix.log 2>&1
        bash run_testcase_v25_baseline_single.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v25_baseline_single.log 2>&1
    ;;
    v26)
        bash run_testcase_v26_baseline_single.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v26_baseline_single.log 2>&1
        bash run_testcase_v26_req_single.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v26_req_single.log 2>&1
    ;;
    s)
#        bash run_testcase_v25_baseline_single.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v25_baseline_single.log 2>&1
        bash run_testcase_v26_req_single.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v26_req_single.log 2>&1
    ;;
    *)
        bash run_testcase_v25_baseline_multi_pd_separation.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v25_baseline_multi_pd_separation.log 2>&1
        bash run_testcase_v25_baseline_multi_pd_mix.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v25_baseline_multi_pd_mix.log 2>&1
        bash run_testcase_v25_baseline_single.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v25_baseline_single.log 2>&1
        bash run_testcase_v26_baseline_single.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v26_baseline_single.log 2>&1
#        bash run_testcase_v26_req_single.sh $sglang_source_path $install_sglang_from_source $image > log/run_testcase_v26_req_single.log 2>&1
    ;;
esac

