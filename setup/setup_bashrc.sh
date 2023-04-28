#!/bin/bash
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

if [ -z "$1" ]
  then echo "No username is supplied: arg1:=username"
  exit
fi

user_bash_USER_HOME=`cat /home/${1}/.bashrc | grep USER_HOME | head -1`
root_bash_USER_HOME=`cat /root/.bashrc | grep USER_HOME | head -1`
change_flag="False"

if [ "$user_bash_USER_HOME" != "export USER_HOME=/home/${1}" ]
  then echo "export USER_HOME=/home/${1}" >> /home/${1}/.bashrc
  echo "\$USER_HOME is updated to /home/${1}/.bashrc"
  change_flag="True"
fi

if [ "$root_bash_USER_HOME" != "export USER_HOME=/home/${1}" ]
  then echo "export USER_HOME=/home/${1}" >> /root/.bashrc
  echo "\$USER_HOME is updated to /root/.bashrc"
  change_flag="True"
fi


autoware_path="$(cd -P /home/${1}/autoware.ai && pwd)"
rubis_ws_path="$(cd -P /home/${1}/rubis_ws && pwd)"

user_bash_rubis_pkg=`cat /home/${1}/.bashrc | grep "source ${rubis_ws_path}/devel/setup.bash" | head -1`
root_bash_rubis_pkg=`cat /root/.bashrc | grep "source ${rubis_ws_path}/devel/setup.bash" | head -1`
user_bash_AUTWOARE=`cat /home/${1}/.bashrc | grep "source ${autoware_path}/devel/setup.bash" | head -1`
root_bash_AUTWOARE=`cat /root/.bashrc | grep "source ${autoware_path}/devel/setup.bash" | head -1`

if [ "$user_bash_AUTWOARE" != "source ${autoware_path}/devel/setup.bash" ]
  then echo "source ${autoware_path}/devel/setup.bash"  >> /home/${1}/.bashrc
  echo "AUTOWAER variable setup is updated to /home/${1}/.bashrc"
  change_flag="True"
fi

if [ "$root_bash_AUTWOARE" != "source ${autoware_path}/devel/setup.bash" ]
  then echo "source ${autoware_path}/devel/setup.bash" >> /root/.bashrc
  echo "AUTOWAER variable setup is updated to /root/.bashrc"
  change_flag="True"
fi


if [ "$user_bash_rubis_pkg" != "source ${rubis_ws_path}/devel/setup.bash" ]
  then echo "source ${rubis_ws_path}/devel/setup.bash" >> /home/${1}/.bashrc
  echo "rubis_pkg variable setup is updated to /home/${1}/.bashrc"
  change_flag="True"
fi

if [ "$root_bash_rubis_pkg" != "source ${rubis_ws_path}/devel/setup.bash" ]
  then echo "source ${rubis_ws_path}/devel/setup.bash" >> /root/.bashrc
  echo "rubis_pkg variable setup is updated to /root/.bashrc"
  change_flag="True"
fi


read -p "Do you want to add hotkeys to bashrc (y/n)?" answer

if [ "$1" = "nvidia" ]
  then lglaunch_dir_name="nvidia"
else
  lglaunch_dir_name="desktop"
fi

alias_user_gb="alias gb='gedit /home/${1}/.bashrc'"
alias_user_sb="alias sb='source /home/${1}/.bashrc'"
alias_root_gb="alias gb='gedit /root/.bashrc'"
alias_root_sb="alias sb='source /root/.bashrc'"
alias_sa="alias sa='source ${autoware_path}/devel/setup.bash'"
alias_sr="alias sa='source ${rubis_ws_path}/devel/setup.bash'"
alias_lglaunch="alias lglaunch='cd /home/${1}/autoware.ai/autoware_files/lgsvl_file/launch/${lglaunch_dir_name}'"

grep_user_gb=`cat /home/${1}/.bashrc | grep "${alias_user_gb}" | head -1`
grep_user_sb=`cat /home/${1}/.bashrc | grep "${alias_user_sb}" | head -1`
grep_user_sa=`cat /home/${1}/.bashrc | grep "${alias_sa}" | head -1`
grep_user_sr=`cat /home/${1}/.bashrc | grep "${alias_sr}" | head -1`
grep_user_lglaunch=`cat /home/${1}/.bashrc | grep "${alias_lglaunch}" | head -1`
grep_root_gb=`cat /root/.bashrc | grep "${alias_root_gb}" | head -1`
grep_root_sb=`cat /root/.bashrc | grep "${alias_root_sb}" | head -1`
grep_root_sa=`cat /root/.bashrc | grep "${alias_sa}" | head -1`
grep_root_sr=`cat /root/.bashrc | grep "${alias_sr}" | head -1`
grep_root_lglaunch=`cat /root/.bashrc | grep "${alias_lglaunch}" | head -1`

case ${answer:0:1} in
    y|Y )
        if [ "$grep_user_gb" != "$alias_user_gb" ]
          then echo "${alias_user_gb}" >> /home/${1}/.bashrc
        fi
        if [ "$grep_user_sb" != "$alias_user_sb" ]
          then echo "${alias_user_sb}" >> /home/${1}/.bashrc
        fi
        if [ "$grep_user_sa" != "$alias_sa" ]
          then echo "${alias_sa}" >> /home/${1}/.bashrc
        fi
        if [ "$grep_user_sr" != "$alias_sr" ]
          then echo "${alias_sr}" >> /home/${1}/.bashrc
        fi               
        if [ "$grep_user_lglaunch" != "$alias_lglaunch" ]
          then echo "${alias_lglaunch}" >> /home/${1}/.bashrc
        fi         

        if [ "$grep_root_gb" != "$alias_root_gb" ]
          then echo "${alias_root_gb}" >> /root/.bashrc
        fi
        if [ "$grep_root_sb" != "$alias_root_sb" ]
          then echo "${alias_root_sb}" >> /root/.bashrc
        fi
        if [ "$grep_root_sa" != "$alias_sa" ]
          then echo "${alias_sa}" >> /root/.bashrc
        fi
        if [ "$grep_root_sr" != "$alias_sr" ]
          then echo "${alias_sr}" >> /root/.bashrc
        fi               
        if [ "$grep_root_lglaunch" != "$alias_lglaunch" ]
          then echo "${alias_lglaunch}" >> /root/.bashrc
        fi                   
    ;;
    * )
        echo "Hotkeys will not be added to bashrc"
    ;;
esac

if [ "$change_flag" = "True" ]
  then echo "[NOTIFIACTION] Please re-open all terminals"
fi