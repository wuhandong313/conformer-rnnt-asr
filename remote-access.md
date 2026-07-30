# 内网服务器远程访问指南

## Gitee 仓库

| 项目 | 详情 |
|------|------|
| **仓库链接** | https://gitee.com/wu3134/conformer-rnnt-asr |
| **用户名** | `wu3134` |
| **私人令牌 (token)** | `f9f331a369d34a8ecaa0abd4dd35acca` |

带凭证的 clone / pull 地址:

```
https://wu3134:f9f331a369d34a8ecaa0abd4dd35acca@gitee.com/wu3134/conformer-rnnt-asr.git
```

### 配置凭证免重复输入

```bash
cd conformer-rnnt-asr
git config credential.helper store   # 明文保存到 ~/.git-credentials
```

之后第一次 pull/push 输入用户名和 token, 之后自动读取。

### 服务器上首次 clone

```bash
cd /test21
git clone https://wu3134:f9f331a369d34a8ecaa0abd4dd35acca@gitee.com/wu3134/conformer-rnnt-asr.git
cd conformer-rnnt-asr
git config credential.helper store
```

## 网络拓扑

```
Mac (外网) ──ZeroTier──> 学校电脑 (10.137.230.170) ──内网──> 服务器 (192.168.19.114:30584)
```

## 关键信息

| 项目 | 详情 |
|------|------|
| **ZeroTier Network ID** | `743993800fc402e5` |
| **ZeroTier 网络名** | gpu |
| **Mac ZeroTier IP** | `10.137.230.14` |
| **学校电脑 (跳板机)** | `10.137.230.170` |
| **学校电脑用户/密码** | `vuser` / `user@123` |
| **内网服务器** | `192.168.19.114` |
| **内网服务器 SSH 端口** | `30584` |
| **内网服务器用户/密码** | `root` / `559454` |
| **内网 Web 服务** | `https://192.168.19.114:32206/` |

## SSH Config

已配置在 `~/.ssh/config`:

```
Host internal
  HostName 192.168.19.114
  Port 30584
  User root
  ProxyCommand sshpass -p 'user@123' ssh -o StrictHostKeyChecking=no -W %h:%p vuser@10.137.230.170
```

## 常用命令

### SSH 登录

```bash
sshpass -p '559454' ssh internal
```

### 文件传输 (scp)

```bash
# 上传文件
sshpass -p '559454' scp ~/本地文件 internal:/test21/

# 下载文件
sshpass -p '559454' scp internal:/test21/文件名 ~/

# 上传文件夹
sshpass -p '559454' scp -r ~/文件夹 internal:/test21/
```

### 浏览器访问内网网页

```bash
# 先建立端口转发
sshpass -p 'user@123' ssh -o StrictHostKeyChecking=no -L 32206:192.168.19.114:32206 vuser@10.137.230.170 -N -f

# 浏览器打开
# https://localhost:32206/
```

### Cyberduck / 传文件 App

```bash
# 先建立隧道 (只需一次)
sshpass -p 'user@123' ssh -o StrictHostKeyChecking=no -L 2222:192.168.19.114:30584 vuser@10.137.230.170 -N -f
```

然后在 Cyberduck 中配置:

| 设置项 | 值 |
|--------|-----|
| 协议 | SFTP |
| 服务器 | localhost |
| 端口 | 2222 |
| 用户名 | root |
| 密码 | 559454 |

### 便捷别名 (可选，添加到 ~/.zshrc)

```bash
alias sshin='sshpass -p '\''559454'\'' ssh internal'
alias scpin='sshpass -p '\''559454'\'' scp'
alias webin='sshpass -p '\''user@123'\'' ssh -o StrictHostKeyChecking=no -L 32206:192.168.19.114:32206 vuser@10.137.230.170 -N -f && echo "打开 https://localhost:32206/"'
alias duckin='sshpass -p '\''user@123'\'' ssh -o StrictHostKeyChecking=no -L 2222:192.168.19.114:30584 vuser@10.137.230.170 -N -f && echo "Cyberduck 连 localhost:2222"'
```

## 重新配置

如果服务器信息变了，改以下几个地方就行:

### 涉及的文件

| 文件 | 作用 |
|------|------|
| `~/.ssh/config` | SSH 登录配置 |
| `~/.zshrc` | 便捷别名 |
| 本文件 | 参考文档 |

### 场景一: 内网服务器 IP / 端口 / 用户 / 密码变了

**步骤:**

**1. 更新 `~/.ssh/config`** — 改 `Host internal` 这一段:

```
Host internal
  HostName <新IP>          # 改这里
  Port <新端口>             # 改这里
  User <新用户名>           # 改这里
  ProxyCommand sshpass -p '<跳板机密码>' ssh -o StrictHostKeyChecking=no -W %h:%p <跳板机用户>@<跳板机ZeroTier IP>
```

**2. 更新便捷别名** (如果配置了的话) — 改 `~/.zshrc`:

```bash
# 把命令里的 IP、端口、密码换成新的
alias sshin='sshpass -p '\''<新密码>'\'' ssh internal'
alias webin='sshpass -p '\''<跳板机密码>'\'' ssh -o StrictHostKeyChecking=no -L <本地端口>:<新IP>:<新Web端口> <跳板机用户>@<跳板机IP> -N -f'
alias duckin='sshpass -p '\''<跳板机密码>'\'' ssh -o StrictHostKeyChecking=no -L 2222:<新IP>:<新SSH端口> <跳板机用户>@<跳板机IP> -N -f'
```

然后执行 `source ~/.zshrc` 生效。

**3. 更新本文件** — 把上面"关键信息"表格里的对应值改掉。

---

### 场景二: 学校电脑变了 (密码 / IP / 用户)

**1. 更新 `~/.ssh/config`** — 改 `ProxyCommand` 那行:

```
ProxyCommand sshpass -p '<新跳板机密码>' ssh -o StrictHostKeyChecking=no -W %h:%p <新用户名>@<新ZeroTier IP>
```

**2. 更新所有别名** — `~/.zshrc` 中所有 `sshpass -p 'user@123'` 改成新密码。

**3. 如果学校电脑是新加入 ZeroTier 的:**

```bash
# 新学校电脑上执行
curl -s https://install.zerotier.com | bash && zerotier-cli join 743993800fc402e5
# 然后去 my.zerotier.com 授权，记下分配的 IP (10.137.230.x)
```

---

### 场景三: ZeroTier Network ID 变了

在新电脑/服务器上重新加入:

```bash
zerotier-cli leave <旧Network ID>
zerotier-cli join <新Network ID>
```

然后更新 `~/.ssh/config` 中 `ProxyCommand` 里的跳板机 IP (ZeroTier 分配的 IP 会变)。

---

### 配置位置速查

```
                   值变了      改哪个文件
                   ──────     ──────────
服务器 IP         →   ~/.ssh/config (HostName), ~/.zshrc, 本文
服务器端口         →   ~/.ssh/config (Port), ~/.zshrc, 本文
服务器密码         →   sshpass 命令中的 -p, ~/.zshrc, 本文
服务器用户         →   ~/.ssh/config (User), ~/.zshrc, 本文
Web 端口           →   ~/.zshrc (别名中的 -L 转发), 本文
跳板机密码         →   ~/.ssh/config (ProxyCommand), ~/.zshrc, 本文
跳板机 ZeroTier IP →   ~/.ssh/config (ProxyCommand), ~/.zshrc, 本文
```

## 故障排查

### SSH 连不上

1. 检查 ZeroTier 是否在线: `zerotier-cli listnetworks` 看 `743993800fc402e5` 状态是否为 OK
2. 检查学校电脑是否在线: `ping 10.137.230.170`
3. 如果学校电脑重启了，SSH 隧道会断开，需要重新建立

### ZeroTier 连不上

学校电脑的 SSH 如果也连不上，可能是 ZeroTier 服务没启动:
- 学校电脑上执行: `sudo systemctl start zerotier-one`

### 内网服务器安装 ZeroTier (不可行)

内网服务器是 K8s 中的 Docker 容器，没有 `/dev/net/tun` 设备，无法安装 ZeroTier。
必须通过学校电脑做跳板访问。
