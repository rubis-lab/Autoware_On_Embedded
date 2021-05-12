# **Xavier Flashing**

## **This will install...**

[L4T 32.4.](https://developer.nvidia.com/embedded/linux-tegra-r32.4.2)3

## **Creating NVIDIA Jetson Xavier image**

[Driver Package: L4T Driver Package (BSP)](https://developer.nvidia.com/embedded/L4T/r32_Release_v4.3/t186ref_release_aarch64/Tegra186_Linux_R32.4.3_aarch64.tbz2) Download required packages

```jsx
$ mkdir -p $HOME/nvidia/docs
$ cd $HOME/nvidia
$ wget https://developer.nvidia.com/embedded/L4T/r32_Release_v4.3/t186ref_release_aarch64/Tegra186_Linux_R32.4.3_aarch64.tbz2
$ wget -O NV_Jetson_Linux_Driver_Package.tar https://developer.nvidia.com/nvidia-jetson-linux-driver-package
$ wget https://developer.nvidia.com/embedded/L4T/r32_Release_v4.3/t186ref_release_aarch64/Tegra_Linux_Sample-Root-Filesystem_R32.4.3_aarch64.tbz2
```

Unpack

```jsx
$ tar -xf NV_Jetson_Linux_Driver_Package.tar -C $HOME/nvidia/docs
$ sudo tar -xf Tegra186_Linux_R32.4.3_aarch64.tbz2
# (3 min)
$ sudo tar -xf Tegra_Linux_Sample-Root-Filesystem_R32.4.3_aarch64.tbz2 -C $HOME/nvidia/Linux_for_Tegra/rootfs
$ cd $HOME/nvidia/Linux_for_Tegra
$ sudo ./apply_binaries.sh
```

## **Download dev tools**

```jsx
$ mkdir -p $HOME/nvidia_rt
$ cd $HOME/nvidia_rt
$ wget -O l4t-gcc-7-3-1-toolchain-64-bit.tar.xz https://developer.nvidia.com/embedded/dlc/l4t-gcc-7-3-1-toolchain-64-bit
$ wget https://developer.nvidia.com/embedded/L4T/r32_Release_v4.2/Sources/T186/public_sources.tbz2
```

## **Prepare build sources**

```jsx
$ cd $HOME/nvidia_rt
$ tar -xf l4t-gcc-7-3-1-toolchain-64-bit.tar.xz
$ tar -xf public_sources.tbz2
$ cd $HOME/nvidia_rt/Linux_for_Tegra/source/public
$ tar -xjf kernel_src.tbz2
```

## **Create an `environment` file for envvars**

```jsx
$ cd $HOME/nvidia_rt
$ vim $HOME/nvidia_rt/rubis_env_rt
#!/bin/sh
# toolchain flags
export BSPTOOLCHAIN=$HOME/nvidia_rt/install/bin
export PATH=${BSPTOOLCHAIN}:${PATH}
# kernel flags
export ARCH=arm64
export CROSS_COMPILE=$HOME/nvidia_rt/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-
export TEGRA_XAVIER_KERNEL_OUT=$HOME/nvidia_rt/tegra-jetson-xavier-kernel
export LOCALVERSION=-tegra
```

## **Export build variables and start compiling**

Test kernel config with `CONFIG_LOCALVERSION="-tegra"` and `CONFIG_PREEMPT_RT_FULL=y` can be found [here](https://github.com/kozyilmaz/nvidia-jetson-rt/raw/master/scripts/jetson-tx2-rt.config)

```jsx
$ cd $HOME/nvidia_rt
$ source rubis_env_rt
$ mkdir -p $TEGRA_XAVIER_KERNEL_OUT
$ cd $HOME/nvidia_rt/Linux_for_Tegra/source/public/kernel/kernel-4.9
# list and apply real-time patches
$ ./scripts/rt-patch.sh apply-patches

# create default config
$ make O=$TEGRA_XAVIER_KERNEL_OUT ARCH=$ARCH tegra_defconfig

# compile rt kernel Image
$ make O=$TEGRA_XAVIER_KERNEL_OUT ARCH=$ARCH Image -j12
# copy the kernel Image and paste to the sample file system Image
$ cp $TEGRA_XAVIER_KERNEL_OUT/arch/arm64/boot/Image $HOME/nvidia/Linux_for_Tegra/kernel/Image

# compile and install kernel modules
$ make -j12 O=$TEGRA_XAVIER_KERNEL_OUT ARCH=$ARCH modules
$ make O=$TEGRA_XAVIER_KERNEL_OUT ARCH=$ARCH modules_install INSTALL_MOD_PATH=$TEGRA_XAVIER_KERNEL_OUT/modules

# copy the kernel modules and paste to the sample file system modules
$ cd $TEGRA_XAVIER_KERNEL_OUT/modules/lib/modules
$ sudo cp -r 4.9.140-rt93-tegra/ $HOME/nvidia/Linux_for_Tegra/rootfs/lib/modules/
```

## **Force Recovery Mode for NVIDIA AGX XAVIER**

1. Power down the device. If connected, remove the AC adapter from the device. The device must be powered OFF, and not in a suspend or sleep state.
2. Connect the Type-C plug on the USB cable to the Recovery (USB Type-C) Port on the device and the other end to an available USB port on the host PC.
3. Connect the power adapter to the device.
4. Press and release the POWER button to power on device.
5. Press and hold the FORCE RECOVERY button(중간버튼): while pressing the FORCE RECOVERY button, press and release the RESET button(맨 왼쪽 버튼); wait two seconds and release the FORCE RECOVERY button.
6. When the device is in recovery mode, **lsusb** command on HOST will list a line of **Nvidia Corp**

# **Flashing NVIDIA AGX XAVIER internal eMMC**

```jsx
# run flash command (5 min)
$ sudo ./flash.sh jetson-xavier mmcblk0p1
```

## **[Xavier] Check Kernel Parameters after Reboot**

```jsx
$ zcat /proc/config.gz |grep CONFIG_HZ
# CONFIG_HZ_PERIODIC is not set
# CONFIG_HZ_100 is not set
CONFIG_HZ_250=y
# CONFIG_HZ_300 is not set
# CONFIG_HZ_1000 is not set
CONFIG_HZ=250

$ zcat /proc/config.gz |grep CONFIG_PREEMPT
CONFIG_PREEMPT_RCU=y
CONFIG_PREEMPT=y
CONFIG_PREEMPT_RT_BASE=y
CONFIG_PREEMPT_LAZY=y
# CONFIG_PREEMPT_NONE is not set
# CONFIG_PREEMPT_VOLUNTARY is not set
# CONFIG_PREEMPT__LL is not set
# CONFIG_PREEMPT_RTB is not set
CONFIG_PREEMPT_RT_FULL=y
CONFIG_PREEMPT_COUNT=y
# CONFIG_PREEMPTIRQ_EVENTS is not set
# CONFIG_PREEMPT_TRACER is not set

nvidia@tegra-ubuntu:~/devel/nvidia-jetson-rt$ uname -a
Linux nvidia 4.9.140-rt93-rt #1 SMP PREEMPT RT Thu Aug 19 17:17:52 KST 2020 aarch64 aarch64 aarch64 GNU/Linux

#check L4T version
$ cat /etc/nv_tegra_release
```

## Migrate rootfs to SSD

[https://medium.com/@ramin.nabati/installing-an-nvme-ssd-drive-on-nvidia-jetson-xavier-37183c948978](https://medium.com/@ramin.nabati/installing-an-nvme-ssd-drive-on-nvidia-jetson-xavier-37183c948978)

⇒위 링크 참고해서 파티션으로 설정 후

파티션은 /mnt에 마운트 해야함. ⇒ 위의 링크에서 /xavier_ssd를 /mnt라고 생각하면 됨.

```jsx
$ sudo mount /dev/nvme0n1p1 /mnt
$ sudo cp /etc/fstab /etc/fstab.bkup
$ sudo gedit /etc/fstab
```

In fstab

```jsx
UUID=<uuid>  /mnt  ext4  defaults  0  2
```

UUID는 disks에서 확인할 수 있음

```jsx
$ sudo umount /mnt
$ sudo mount -a
```

```jsx
$ git clone https://github.com/jetsonhacks/rootOnNVMe.git
$ cd rootOnNVMe
$ ./copy-rootfs-ssd.sh
$ ./setup-service.sh
$ sudo reboot
```