import discord
from discord.ext import commands
from discord import app_commands # 用于斜杠命令
import os
import aiohttp
import json
import logging
from collections import deque
import asyncio
import re # 用于清理用户名

# --- 提前设置日志记录 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 配置 ---
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
logger.info(f"Initializing with DeepSeek Model: {DEEPSEEK_MODEL}")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10")) # 总轮数，实际问答为 MAX_HISTORY / 2
SPLIT_MESSAGE_DELAY = float(os.getenv("SPLIT_MESSAGE_DELAY", "0.3")) # 分割消息发送延迟（秒）
# --- 新增：配置管理员/管理角色 ID (可选, 逗号分隔) ---
admin_ids_str = os.getenv("ADMIN_ROLE_IDS", "")
ADMIN_ROLE_IDS = [int(role_id) for role_id in admin_ids_str.split(",") if role_id.strip().isdigit()]
# --- 新增：用于识别机器人创建的频道的名称前缀 ---
PRIVATE_CHANNEL_PREFIX = "deepseek-"

# --- DeepSeek API 请求函数 ---
async def get_deepseek_response(session, api_key, model, messages):
    """异步调用 DeepSeek API，处理 reasoning_content 和 content"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        # 注意：根据文档，reasoner 模型不支持 temperature, top_p 等参数
    }

    logger.info(f"Sending request to DeepSeek API using model '{model}' with {len(messages)} messages.")
    # logger.debug(f"Request payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as response:
            # 尝试直接读取文本以防 JSON 解析失败时也能看到内容
            raw_response_text = await response.text()
            # logger.debug(f"Raw response text: {raw_response_text[:500]}")

            try:
                response_data = json.loads(raw_response_text)
            except json.JSONDecodeError:
                 logger.error(f"Failed to decode JSON response from DeepSeek API. Status: {response.status}. Raw text started with: {raw_response_text[:500]}...")
                 return None, f"抱歉，无法解析 DeepSeek API 的响应 (状态码 {response.status})。"

            # logger.debug(f"Parsed response data: {json.dumps(response_data, indent=2, ensure_ascii=False)}")

            if response.status == 200:
                if response_data.get("choices") and len(response_data["choices"]) > 0:
                    message_data = response_data["choices"][0].get("message", {})
                    reasoning_content = message_data.get("reasoning_content")
                    final_content = message_data.get("content")
                    usage = response_data.get("usage")

                    full_response_for_discord = ""
                    if reasoning_content:
                        full_response_for_discord += f"🤔 **思考过程:**\n```\n{reasoning_content.strip()}\n```\n\n"
                    if final_content:
                        full_response_for_discord += f"💬 **最终回答:**\n{final_content.strip()}"
                    if not final_content and reasoning_content: # 如果只有思维链
                        full_response_for_discord = reasoning_content.strip()

                    if not full_response_for_discord: # 如果两者都没有
                        logger.error("DeepSeek API response missing both reasoning_content and content.")
                        return None, "抱歉，DeepSeek API 返回的数据似乎不完整。"

                    logger.info(f"Successfully received response from DeepSeek. Usage: {usage}")
                    return full_response_for_discord.strip(), final_content.strip() if final_content else None
                else:
                    logger.error(f"DeepSeek API response missing 'choices': {response_data}")
                    return None, f"抱歉，DeepSeek API 返回了意外的结构：{response_data}"

            elif response.status == 400:
                 error_message = response_data.get("error", {}).get("message", "未知 400 错误")
                 logger.error(f"DeepSeek API error (Status 400 - Bad Request): {error_message}")
                 logger.warning("A 400 error might be caused by including 'reasoning_content' in the input messages. Check conversation history logic.")
                 return None, f"请求错误 (400): {error_message}\n(请检查是否意外将思考过程加入到了请求历史中)"
            else:
                error_message = response_data.get("error", {}).get("message", f"未知错误 (状态码 {response.status})")
                logger.error(f"DeepSeek API error (Status {response.status}): {error_message}. Response: {raw_response_text[:500]}")
                return None, f"抱歉，调用 DeepSeek API 时出错 (状态码 {response.status}): {error_message}"

    except aiohttp.ClientConnectorError as e:
        logger.error(f"Network connection error: {e}")
        return None, f"抱歉，无法连接到 DeepSeek API：{e}"
    except asyncio.TimeoutError:
        logger.error("Request to DeepSeek API timed out.")
        return None, "抱歉，连接 DeepSeek API 超时。"
    except Exception as e:
        logger.exception("An unexpected error occurred during DeepSeek API call.")
        return None, f"抱歉，处理 DeepSeek 请求时发生未知错误: {e}"

# --- Discord 机器人设置 ---
intents = discord.Intents.default()
intents.messages = True       # 需要读取消息
intents.message_content = True # 需要读取非@或命令的消息内容 (在私有频道中)
intents.guilds = True         # 需要访问服务器信息以创建频道

# --- 改用 commands.Bot ---
bot = commands.Bot(command_prefix="!", intents=intents) # command_prefix 不会用到，但必须提供

# 对话历史 (key 是私有频道的 ID)
conversation_history = {}

# --- setup_hook 同步斜杠命令和打印配置 ---
@bot.event
async def setup_hook():
    logger.info("Running setup_hook...")
    logger.info(f'--- Bot Configuration ---')
    logger.info(f'Default/Current DeepSeek Model: {DEEPSEEK_MODEL}')
    logger.info(f'Max Conversation History Turn: {MAX_HISTORY}')
    logger.info(f'Split Message Delay: {SPLIT_MESSAGE_DELAY}s')
    logger.info(f'Admin Role IDs: {ADMIN_ROLE_IDS}')
    logger.info(f'Private Channel Prefix: {PRIVATE_CHANNEL_PREFIX}')
    logger.info(f'Discord.py Version: {discord.__version__}')
    logger.info(f'-------------------------')
    try:
        # 同步全局命令。如果只想在特定测试服务器同步，请取消注释并设置 TEST_GUILD_ID
        # test_guild_id = os.getenv("TEST_GUILD_ID")
        # if test_guild_id:
        #     guild = discord.Object(id=int(test_guild_id))
        #     bot.tree.copy_global_to(guild=guild)
        #     synced = await bot.tree.sync(guild=guild)
        #     logger.info(f"Synced {len(synced)} commands to guild {test_guild_id}.")
        # else:
        #     synced = await bot.tree.sync()
        #     logger.info(f"Synced {len(synced)} commands globally.")
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} commands globally.")
    except Exception as e:
        logger.exception(f"Failed to sync slash commands: {e}")

@bot.event
async def on_ready():
    logger.info(f'机器人已登录为 {bot.user}')
    print(f"Logged in as {bot.user} ({bot.user.id})")
    print("Bot is ready and slash commands should be available.")

# --- 斜杠命令：创建新的聊天频道 ---
@bot.tree.command(name="new_chat", description="创建一个与 DeepSeek 的私密聊天频道")
@app_commands.checks.has_permissions(send_messages=True) # 基本权限检查
async def new_chat(interaction: discord.Interaction):
    """处理 /new_chat 命令"""
    guild = interaction.guild
    user = interaction.user

    if not guild:
        await interaction.response.send_message("此命令只能在服务器内使用。", ephemeral=True)
        return

    # 检查机器人是否有创建频道的权限
    bot_member = guild.get_member(bot.user.id)
    if not bot_member or not bot_member.guild_permissions.manage_channels:
        await interaction.response.send_message("机器人缺少“管理频道”权限，无法创建聊天频道。", ephemeral=True)
        logger.warning(f"Missing 'Manage Channels' permission in guild {guild.id} or bot member not found.")
        return

    # 清理用户名
    clean_user_name = re.sub(r'[^\w-]', '', user.display_name.replace(' ', '-')).lower()
    if not clean_user_name: clean_user_name = "user" # 防止用户名全是特殊字符
    channel_name = f"{PRIVATE_CHANNEL_PREFIX}{clean_user_name}-{user.id % 10000}" # 部分ID防重名

    # 权限覆盖
    overwrites = {
        guild.default_role: discord.PermissionOverwrite(read_messages=False), # @everyone 不可见
        user: discord.PermissionOverwrite(read_messages=True, send_messages=True, view_channel=True),
        bot_member: discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, manage_channels=True)
    }
    for role_id in ADMIN_ROLE_IDS:
        role = guild.get_role(role_id)
        if role:
            overwrites[role] = discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, view_channel=True)
        else:
             logger.warning(f"Admin role ID {role_id} not found in guild {guild.id}")

    try:
        # 告知用户正在创建，防止交互超时
        await interaction.response.send_message(f"正在为你创建私密聊天频道 **{channel_name}** ...", ephemeral=True)

        # 创建频道 (可以考虑放在特定分类下)
        category = discord.utils.find(lambda c: c.name.lower() == "deepseek chats", guild.categories) # 尝试寻找分类
        if category and category.permissions_for(bot_member).manage_channels:
             new_channel = await guild.create_text_channel(channel_name, overwrites=overwrites, category=category)
             logger.info(f"Created channel {new_channel.name} in category '{category.name}'")
        else:
             if category: logger.warning(f"Found category 'DeepSeek Chats' but lack permissions or it's not a category channel. Creating in default location.")
             else: logger.info("Category 'DeepSeek Chats' not found, creating in default location.")
             new_channel = await guild.create_text_channel(channel_name, overwrites=overwrites)

        logger.info(f"Successfully created private channel {new_channel.name} ({new_channel.id}) for user {user.name} ({user.id}) in guild {guild.id}")

        # 在新频道发送欢迎消息
        await new_channel.send(
            f"你好 {user.mention}！\n"
            f"欢迎来到你的 DeepSeek 私密聊天频道 (使用模型: **{DEEPSEEK_MODEL}**)。\n"
            f"直接在此输入你的问题即可开始对话。\n"
            f"对话历史最多保留 **{MAX_HISTORY // 2}** 轮问答。\n"
            f"当你完成后，可以在此频道使用 `/close_chat` 命令来关闭它。"
        )

        # 编辑初始响应，提供频道链接 (followup 用于在 ephemeral 消息后添加可见内容)
        await interaction.followup.send(f"你的私密聊天频道已创建：{new_channel.mention}", ephemeral=True)

    except discord.Forbidden:
        logger.error(f"Permission error (Forbidden) while trying to create/configure channel '{channel_name}' in guild {guild.id}.")
        try: await interaction.followup.send("创建频道失败：机器人权限不足。请检查机器人角色是否有“管理频道”权限，并且没有被频道分类覆盖。", ephemeral=True)
        except discord.NotFound: pass
    except discord.HTTPException as e:
        logger.error(f"HTTP error while creating channel '{channel_name}' in guild {guild.id}: {e}")
        try: await interaction.followup.send(f"创建频道时发生网络错误：{e}", ephemeral=True)
        except discord.NotFound: pass
    except Exception as e:
        logger.exception(f"Unexpected error during /new_chat command for user {user.id} in guild {guild.id}")
        try: await interaction.followup.send(f"创建频道时发生未知错误。", ephemeral=True)
        except discord.NotFound: pass

# --- 斜杠命令：关闭聊天频道 ---
@bot.tree.command(name="close_chat", description="关闭当前的 DeepSeek 私密聊天频道")
@app_commands.checks.has_permissions(send_messages=True) # 基本权限检查
async def close_chat(interaction: discord.Interaction):
    """处理 /close_chat 命令"""
    channel = interaction.channel
    user = interaction.user
    guild = interaction.guild

    if not isinstance(channel, discord.TextChannel) or not channel.name.startswith(PRIVATE_CHANNEL_PREFIX):
        await interaction.response.send_message("此命令只能在 DeepSeek 私密聊天频道中使用。", ephemeral=True)
        return

    # 简单的权限检查：允许创建者或拥有管理员角色的人关闭
    # 注意：这种从频道名称解析创建者ID的方法不是最可靠的，但对于这个场景够用
    try:
        channel_creator_id_part = channel.name.split('-')[-1]
        # (这里可以加更严格的检查，例如确保分割出的确实是数字等)
    except IndexError:
        channel_creator_id_part = None #无法从名称中解析

    is_admin = any(role.id in ADMIN_ROLE_IDS for role in user.roles) if isinstance(user, discord.Member) else False
    
    # 如果无法从频道名获取ID，只允许管理员关闭
    # if channel_creator_id_part is None and not is_admin:
    #      await interaction.response.send_message("无法确定频道创建者，只有管理员可以关闭此频道。", ephemeral=True)
    #      return
    # elif channel_creator_id_part and user.id != int(channel_creator_id_part) and not is_admin:
    #      await interaction.response.send_message("只有频道创建者或管理员可以关闭此频道。", ephemeral=True)
    #      return
    # (暂时取消严格的关闭权限检查，让任何人都能在频道内关闭，如果需要再加回来)

    try:
        # 先确认交互
        await interaction.response.send_message(f"请求收到！频道 {channel.mention} 将在几秒后关闭...", ephemeral=True)
        # 在频道内发送公开消息
        await channel.send(f"此聊天频道由 {user.mention} 请求关闭，将在 5 秒后删除。")

        logger.info(f"User {user.name} ({user.id}) initiated closure of channel {channel.name} ({channel.id})")

        # 从内存中移除历史记录
        if channel.id in conversation_history:
            del conversation_history[channel.id]
            logger.info(f"Removed conversation history for channel {channel.id}")

        # 延迟后删除频道
        await asyncio.sleep(5)
        await channel.delete(reason=f"Closed by user {user.name} ({user.id}) via /close_chat")
        logger.info(f"Deleted channel {channel.name} ({channel.id})")

    except discord.Forbidden:
        logger.error(f"Permission error (Forbidden) while trying to delete channel {channel.id}.")
        await interaction.followup.send("关闭频道失败：机器人缺少“管理频道”权限。", ephemeral=True)
    except discord.NotFound:
        logger.warning(f"Channel {channel.id} not found during deletion, likely already deleted.")
    except discord.HTTPException as e:
        logger.error(f"HTTP error while deleting channel {channel.id}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during /close_chat command for channel {channel.id}")

# --- on_message 事件处理 (核心交互逻辑) ---
@bot.event
async def on_message(message: discord.Message):
    """收到消息时执行，只处理来自特定私有频道的消息"""
    # 忽略自己、其他机器人或 Webhook
    if message.author == bot.user or message.author.bot:
        return

    # 只处理特定前缀的文本频道
    if not isinstance(message.channel, discord.TextChannel) or not message.channel.name.startswith(PRIVATE_CHANNEL_PREFIX):
        return

    # 忽略可能的命令调用（虽然不应该在这里用）
    # if message.content.startswith("!"): return

    user_prompt = message.content.strip()
    if not user_prompt: # 忽略空消息
        return

    channel_id = message.channel.id
    logger.info(f"Handling message in private channel {message.channel.name} ({channel_id}) from {message.author}: '{user_prompt[:50]}...'")

    # 获取或创建历史
    if channel_id not in conversation_history:
        conversation_history[channel_id] = deque(maxlen=MAX_HISTORY)
        logger.info(f"Initialized history for channel {channel_id}")

    history_deque = conversation_history[channel_id]

    # 准备 API 请求消息列表
    api_messages = []
    for msg in history_deque:
        # 确保只发送必要的 role 和 content
        api_messages.append({"role": msg.get("role"), "content": msg.get("content")})
    api_messages.append({"role": "user", "content": user_prompt})

    # 调用 API 并处理响应
    try:
      async with message.channel.typing():
          async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session: # 增加超时时间
              response_for_discord, response_for_history = await get_deepseek_response(session, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, api_messages)

      if response_for_discord: # 成功获取响应
          if response_for_history: # 确保有内容可添加到历史
                history_deque.append({"role": "user", "content": user_prompt})
                history_deque.append({"role": "assistant", "content": response_for_history})
                logger.debug(f"Added user & assistant turn to history for channel {channel_id}. History size: {len(history_deque)}")
          else:
                logger.warning(f"Response for channel {channel_id} lacked final content. Not adding assistant turn.")

          # 发送并处理长消息分割
          if len(response_for_discord) <= 2000:
              await message.channel.send(response_for_discord)
          else:
              logger.warning(f"Response for channel {channel_id} is too long ({len(response_for_discord)} chars), splitting.")
              parts = []
              current_pos = 0
              while current_pos < len(response_for_discord):
                    cut_off = min(current_pos + 1990, len(response_for_discord))
                    # 优先找最后一个换行符
                    split_index = response_for_discord.rfind('\n', current_pos, cut_off)
                    # 找不到或太近，找最后一个空格
                    if split_index == -1 or split_index <= current_pos:
                        space_split_index = response_for_discord.rfind(' ', current_pos, cut_off)
                        if space_split_index != -1 and space_split_index > current_pos:
                            split_index = space_split_index
                        else: # 硬切
                            split_index = cut_off
                    # (简化版代码块保护，可以根据需要增强)
                    chunk_to_check = response_for_discord[current_pos:split_index]
                    if "```" in chunk_to_check and chunk_to_check.count("```") % 2 != 0:
                        fallback_split = response_for_discord.rfind('\n', current_pos, split_index - 1)
                        if fallback_split != -1 and fallback_split > current_pos:
                             split_index = fallback_split

                    parts.append(response_for_discord[current_pos:split_index])
                    current_pos = split_index
                    # 跳过分割处的空白符
                    while current_pos < len(response_for_discord) and response_for_discord[current_pos].isspace():
                        current_pos += 1

              for i, part in enumerate(parts):
                  if not part.strip(): continue
                  if i > 0 and SPLIT_MESSAGE_DELAY > 0:
                       await asyncio.sleep(SPLIT_MESSAGE_DELAY)
                  await message.channel.send(part.strip())

      elif response_for_history: # API 返回错误信息
            await message.channel.send(f"抱歉，处理你的请求时发生错误：\n{response_for_history}")
      else: # 未知错误
          logger.error(f"Received unexpected None values from API call for channel {channel_id}.")
          await message.channel.send("抱歉，与 DeepSeek API 通信时发生未知问题。")

    except discord.Forbidden:
         logger.warning(f"Missing permissions (Forbidden) to send message in channel {channel_id}. Maybe channel deleted or permissions changed.")
    except discord.HTTPException as e:
         logger.error(f"Failed to send message to channel {channel_id} due to HTTPException: {e}")
    except Exception as e:
        logger.exception(f"An error occurred in on_message handler for channel {channel_id}")
        try:
            await message.channel.send(f"处理你的消息时发生内部错误，请稍后再试或联系管理员。错误：{e}")
        except Exception:
            logger.error(f"Could not even send the error message to channel {channel_id}.")


# --- 运行 Bot ---
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        logger.critical("错误：未找到 Discord Bot Token (环境变量 DISCORD_BOT_TOKEN)")
        exit("请设置 DISCORD_BOT_TOKEN 环境变量")
    if not DEEPSEEK_API_KEY:
        logger.critical("错误：未找到 DeepSeek API Key (环境变量 DEEPSEEK_API_KEY)")
        exit("请设置 DEEPSEEK_API_KEY 环境变量")

    try:
        logger.info("尝试启动 Discord 机器人 (commands.Bot)...")
        bot.run(DISCORD_BOT_TOKEN, log_handler=None) # 禁用 discord.py 的默认日志处理，因为我们已经配置了 logging
    except discord.LoginFailure:
        logger.critical("Discord Bot Token 无效，登录失败。请检查你的 Token。")
    except discord.PrivilegedIntentsRequired as e:
        logger.critical(f"必需的 Intents 未启用！请在 Discord Developer Portal -> Bot -> Privileged Gateway Intents 中开启 Message Content Intent。错误: {e}")
    except Exception as e:
        logger.exception("启动机器人时发生未捕获的错误。")