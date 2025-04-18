import discord
from discord.ext import commands
from discord import app_commands # 用于斜杠命令
from discord.ui import View, Button, button # 用于按钮和视图
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
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10")) # 总轮数 (用户+机器人)
SPLIT_MESSAGE_DELAY = float(os.getenv("SPLIT_MESSAGE_DELAY", "0.3")) # 分割消息发送延迟（秒）
admin_ids_str = os.getenv("ADMIN_ROLE_IDS", "") # 逗号分隔的角色ID
ADMIN_ROLE_IDS = [int(role_id) for role_id in admin_ids_str.split(",") if role_id.strip().isdigit()]
PRIVATE_CHANNEL_PREFIX = "deepseek-" # 用于识别私密频道

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
    }
    logger.info(f"Sending request to DeepSeek API using model '{model}' with {len(messages)} messages.")
    try:
        # 增加请求超时时间 (例如 5 分钟)
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=300) as response:
            raw_response_text = await response.text()
            try:
                response_data = json.loads(raw_response_text)
            except json.JSONDecodeError:
                 logger.error(f"Failed to decode JSON response from DeepSeek API. Status: {response.status}. Raw text: {raw_response_text[:500]}...")
                 return None, f"抱歉，无法解析 DeepSeek API 的响应 (状态码 {response.status})。"

            if response.status == 200:
                 if response_data.get("choices") and len(response_data["choices"]) > 0:
                    message_data = response_data["choices"][0].get("message", {})
                    reasoning_content = message_data.get("reasoning_content")
                    final_content = message_data.get("content")
                    usage = response_data.get("usage") # 获取 token 使用情况

                    # 组合输出字符串
                    full_response_for_discord = ""
                    if reasoning_content:
                        full_response_for_discord += f"🤔 **思考过程:**\n```\n{reasoning_content.strip()}\n```\n\n"
                    if final_content:
                        full_response_for_discord += f"💬 **最终回答:**\n{final_content.strip()}"
                    # 处理只有思维链的情况
                    if not final_content and reasoning_content:
                         full_response_for_discord = reasoning_content.strip()
                    # 处理两者都没有的情况
                    if not full_response_for_discord:
                         logger.error("DeepSeek API response missing both reasoning_content and content.")
                         return None, "抱歉，DeepSeek API 返回的数据似乎不完整。"

                    logger.info(f"Successfully received response from DeepSeek. Usage: {usage}")
                    # 返回用于显示的完整内容 和 用于历史记录的最终内容
                    return full_response_for_discord.strip(), final_content.strip() if final_content else None
                 else: # choices 为空
                     logger.error(f"DeepSeek API response missing 'choices': {response_data}")
                     return None, f"抱歉，DeepSeek API 返回了意外的结构：{response_data}"
            else: # 处理非200的HTTP状态码
                error_message = response_data.get("error", {}).get("message", f"未知错误 (状态码 {response.status})")
                logger.error(f"DeepSeek API error (Status {response.status}): {error_message}. Response: {raw_response_text[:500]}")
                if response.status == 400: # 特别提示400错误
                    logger.warning("A 400 error might be caused by including 'reasoning_content' in the input messages.")
                    error_message += "\n(提示: 错误 400 通常因为请求格式错误)"
                return None, f"抱歉，调用 DeepSeek API 时出错 (状态码 {response.status}): {error_message}"
    # 处理网络和超时等异常
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
intents.messages = True
intents.message_content = True
intents.guilds = True
bot = commands.Bot(command_prefix="!", intents=intents) # command_prefix 实际上没用到

# 对话历史 (字典: {channel_id: deque})
conversation_history = {}

# --- 创建按钮视图 ---
class CreateChatView(View):
    # 注意：目前是非持久化视图，机器人重启后按钮会失效
    # 如果需要持久化，需要设置 timeout=None 并 在 setup_hook 中 bot.add_view()
    @button(label="创建私密聊天", style=discord.ButtonStyle.primary, emoji="💬", custom_id="create_deepseek_chat_button")
    async def create_chat_button_callback(self, interaction: discord.Interaction, button_obj: Button):
        """处理“创建私密聊天”按钮点击"""
        guild = interaction.guild
        user = interaction.user
        # 在回调中获取最新的 bot member 对象
        bot_member = guild.get_member(bot.user.id) if guild else None

        if not guild or not bot_member:
            await interaction.response.send_message("无法获取服务器或机器人信息。", ephemeral=True)
            return

        # 检查权限
        if not bot_member.guild_permissions.manage_channels:
            await interaction.response.send_message("机器人缺少“管理频道”权限，无法创建聊天频道。", ephemeral=True)
            logger.warning(f"Button Click: Missing 'Manage Channels' permission in guild {guild.id}")
            return

        # 创建频道名称
        clean_user_name = re.sub(r'[^\w-]', '', user.display_name.replace(' ', '-')).lower()
        if not clean_user_name: clean_user_name = "user" # 防止用户名是空的或只有特殊字符
        channel_name = f"{PRIVATE_CHANNEL_PREFIX}{clean_user_name}-{user.id % 10000}" # 添加部分用户ID避免重名

        # 设置权限覆盖
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(read_messages=False), # @everyone 不可见
            user: discord.PermissionOverwrite(read_messages=True, send_messages=True, view_channel=True), # 用户可见可写
            bot_member: discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, manage_channels=True) # 机器人所需权限
        }
        # 添加管理员角色权限
        for role_id in ADMIN_ROLE_IDS:
            role = guild.get_role(role_id)
            if role:
                overwrites[role] = discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, view_channel=True)
            else:
                 logger.warning(f"Admin role ID {role_id} not found in guild {guild.id}")

        try:
            # 先响应交互，避免超时
            await interaction.response.send_message(f"正在为你创建私密聊天频道 **{channel_name}** ...", ephemeral=True)

            # 尝试找到或创建分类
            category_name = "DeepSeek Chats"
            category = discord.utils.find(lambda c: c.name.lower() == category_name.lower(), guild.categories)
            target_category = None
            if category and isinstance(category, discord.CategoryChannel) and category.permissions_for(bot_member).manage_channels:
                 target_category = category
                 logger.info(f"Found suitable category '{category.name}' for new channel.")
            else:
                 # 如果找不到或无权限，可以选择创建分类或放在顶层
                 logger.warning(f"Category '{category_name}' not found or bot lacks permissions in it. Creating channel in default location.")
                 # 如果需要自动创建分类:
                 # try:
                 #    category_overwrites = {guild.default_role: discord.PermissionOverwrite(read_messages=False)} # 分类也隐藏
                 #    target_category = await guild.create_category(category_name, overwrites=category_overwrites)
                 #    logger.info(f"Created new category '{category_name}'")
                 # except discord.Forbidden:
                 #    logger.error("Failed to create category: Missing permissions.")
                 # except Exception as cat_e:
                 #    logger.exception(f"Error creating category: {cat_e}")

            # 创建文本频道
            new_channel = await guild.create_text_channel(channel_name, overwrites=overwrites, category=target_category)
            logger.info(f"Successfully created private channel {new_channel.name} ({new_channel.id}) for user {user.name} ({user.id})")

            # 在新频道发送欢迎消息
            welcome_message = (
                f"你好 {user.mention}！\n"
                f"欢迎来到你的 DeepSeek 私密聊天频道 (使用模型: **{DEEPSEEK_MODEL}**)。\n"
                f"直接在此输入你的问题即可开始对话。\n"
                f"对话历史最多保留 **{MAX_HISTORY // 2}** 轮问答。\n"
                f"当你完成后，可以在此频道使用 `/close_chat` 命令来关闭它。"
            )
            await new_channel.send(welcome_message)

            # 使用 followup 发送频道链接给点击者
            await interaction.followup.send(f"你的私密聊天频道已创建：{new_channel.mention}", ephemeral=True)

        # 处理各种可能的错误
        except discord.Forbidden:
            logger.error(f"Button Click: Permission error (Forbidden) for user {user.id} creating channel '{channel_name}'.")
            try: await interaction.followup.send("创建频道失败：机器人权限不足。", ephemeral=True)
            except discord.NotFound: pass # 交互可能已经超时
        except discord.HTTPException as e:
            logger.error(f"Button Click: HTTP error for user {user.id} creating channel '{channel_name}': {e}")
            try: await interaction.followup.send(f"创建频道时发生网络错误。", ephemeral=True)
            except discord.NotFound: pass
        except Exception as e:
            logger.exception(f"Button Click: Unexpected error for user {user.id} creating channel")
            try: await interaction.followup.send(f"创建频道时发生未知错误。", ephemeral=True)
            except discord.NotFound: pass

# --- setup_hook: 启动时运行, 同步命令 ---
@bot.event
async def setup_hook():
    logger.info("Running setup_hook...")
    # 打印配置信息
    logger.info(f'--- Bot Configuration ---')
    logger.info(f'Default/Current DeepSeek Model: {DEEPSEEK_MODEL}')
    logger.info(f'Max Conversation History Turn: {MAX_HISTORY}')
    logger.info(f'Split Message Delay: {SPLIT_MESSAGE_DELAY}s')
    logger.info(f'Admin Role IDs: {ADMIN_ROLE_IDS}')
    logger.info(f'Private Channel Prefix: {PRIVATE_CHANNEL_PREFIX}')
    logger.info(f'Discord.py Version: {discord.__version__}')
    logger.info(f'-------------------------')
    # 同步斜杠命令
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash commands globally: {[c.name for c in synced]}")
    except Exception as e:
        logger.exception(f"Failed to sync slash commands: {e}")

# --- on_ready: 机器人准备就绪事件 ---
@bot.event
async def on_ready():
    logger.info(f'机器人已登录为 {bot.user}')
    print(f"Logged in as {bot.user} ({bot.user.id})")
    print("Bot is ready and functional.")

# --- 斜杠命令：/setup_panel ---
@bot.tree.command(name="setup_panel", description="发送一个包含'创建聊天'按钮的消息到当前频道")
async def setup_panel(interaction: discord.Interaction, message_content: str = "点击下面的按钮开始与 DeepSeek 的私密聊天："):
    """发送包含创建聊天按钮的消息，根据频道类型检查权限"""
    channel = interaction.channel
    user = interaction.user # interaction.user 可能是 User 或 Member 对象
    if not interaction.guild:
        await interaction.response.send_message("此命令只能在服务器频道中使用。", ephemeral=True)
        return

    # 条件权限检查
    is_private_chat_channel = isinstance(channel, discord.TextChannel) and channel.name.startswith(PRIVATE_CHANNEL_PREFIX)
    can_execute = False

    if is_private_chat_channel:
        # 在私密频道，允许内部人员执行 (通常是创建者或有权限的管理员)
        can_execute = True
        logger.info(f"User {user} executing /setup_panel in private channel {channel.name}. Allowed.")
    elif isinstance(user, discord.Member) and user.guild_permissions.manage_guild:
        # 在其他频道，需要“管理服务器”权限
        can_execute = True
        logger.info(f"User {user} executing /setup_panel in public channel {channel.name}. Allowed (has manage_guild).")
    else:
        # 在其他频道且无权限
        logger.warning(f"User {user} trying to execute /setup_panel in public channel {channel.name} without manage_guild permission. Denied.")
        await interaction.response.send_message("你需要在非私密频道中使用此命令时拥有“管理服务器”权限。", ephemeral=True)
        return # 阻止执行

    if can_execute:
        try:
            view = CreateChatView() # 创建新的视图实例
            await channel.send(message_content, view=view)
            await interaction.response.send_message("创建聊天按钮面板已发送！", ephemeral=True)
            logger.info(f"User {user} successfully deployed the create chat panel in channel {channel.id}")
        except discord.Forbidden:
             logger.error(f"Failed to send setup panel in {channel.id}: Missing permissions.")
             try: await interaction.followup.send("发送失败：机器人缺少在此频道发送消息或添加组件的权限。", ephemeral=True)
             except discord.NotFound: pass # 交互可能已超时
        except Exception as e:
            logger.exception(f"Failed to send setup panel in {channel.id}")
            try: await interaction.followup.send(f"发送面板时发生错误：{e}", ephemeral=True)
            except discord.NotFound: pass

# --- 斜杠命令：/clear_history ---
@bot.tree.command(name="clear_history", description="清除当前私密聊天频道的对话历史")
async def clear_history(interaction: discord.Interaction):
    """处理 /clear_history 命令"""
    channel = interaction.channel
    user = interaction.user
    # 检查是否在正确的频道类型中
    if not isinstance(channel, discord.TextChannel) or not channel.name.startswith(PRIVATE_CHANNEL_PREFIX):
        await interaction.response.send_message("此命令只能在 DeepSeek 私密聊天频道中使用。", ephemeral=True)
        return

    channel_id = channel.id
    # 清除历史记录
    if channel_id in conversation_history:
        try:
            conversation_history[channel_id].clear() # 清空 deque
            logger.info(f"User {user.name} ({user.id}) cleared history for channel {channel.name} ({channel_id})")
            await interaction.response.send_message("当前频道的对话历史已清除。", ephemeral=False) # 公开确认
        except Exception as e:
            logger.exception(f"Error clearing history for channel {channel_id}")
            await interaction.response.send_message(f"清除历史时发生错误：{e}", ephemeral=True)
    else:
        # 如果内存中没有这个频道的历史记录
        logger.warning(f"User {user.name} ({user.id}) tried to clear history for channel {channel_id}, but no history found.")
        await interaction.response.send_message("未找到当前频道的历史记录（可能从未在此对话或机器人已重启）。", ephemeral=True)

# --- 斜杠命令：/help ---
@bot.tree.command(name="help", description="显示机器人使用帮助")
async def help_command(interaction: discord.Interaction):
    """处理 /help 命令"""
    embed = discord.Embed(
        title="DeepSeek 机器人帮助",
        description=f"你好！我是使用 DeepSeek API ({DEEPSEEK_MODEL}) 的聊天机器人。",
        color=discord.Color.purple() # 可以换个颜色
    )
    embed.add_field(name="如何开始聊天", value="点击管理员放置的 **“创建私密聊天”** 按钮，我会为你创建一个专属频道。", inline=False)
    embed.add_field(name="在私密频道中", value=f"• 直接输入问题进行对话。\n• 我会记住最近 **{MAX_HISTORY // 2}** 轮的问答。", inline=False)
    embed.add_field(name="可用命令", value="`/help`: 显示此帮助信息。\n`/clear_history`: (私密频道内) 清除当前对话历史。\n`/close_chat`: (私密频道内) 关闭当前私密频道。\n`/setup_panel`: (管理员或私密频道内) 发送创建按钮面板。", inline=False)
    embed.set_footer(text=f"当前模型: {DEEPSEEK_MODEL} | 由 discord.py 和 DeepSeek 驱动")
    await interaction.response.send_message(embed=embed, ephemeral=True) # 帮助信息设为仅自己可见

# --- 斜杠命令：/close_chat ---
@bot.tree.command(name="close_chat", description="关闭当前的 DeepSeek 私密聊天频道")
async def close_chat(interaction: discord.Interaction):
    """处理 /close_chat 命令"""
    channel = interaction.channel
    user = interaction.user
    # 再次检查频道类型
    if not isinstance(channel, discord.TextChannel) or not channel.name.startswith(PRIVATE_CHANNEL_PREFIX):
        await interaction.response.send_message("此命令只能在 DeepSeek 私密聊天频道中使用。", ephemeral=True)
        return

    try:
        # 先响应交互
        await interaction.response.send_message(f"请求收到！频道 {channel.mention} 将在几秒后关闭...", ephemeral=True)
        # 在频道内公开通知
        await channel.send(f"此聊天频道由 {user.mention} 请求关闭，将在 5 秒后删除。")
        logger.info(f"User {user.name} ({user.id}) initiated closure of channel {channel.name} ({channel.id})")

        # 从内存清除历史
        if channel.id in conversation_history:
            del conversation_history[channel_id]
            logger.info(f"Removed conversation history for channel {channel.id}")

        # 延迟后删除频道
        await asyncio.sleep(5)
        await channel.delete(reason=f"Closed by user {user.name} ({user.id}) via /close_chat")
        logger.info(f"Deleted channel {channel.name} ({channel.id})")

    except discord.Forbidden:
        logger.error(f"Permission error (Forbidden) while trying to delete channel {channel.id}.")
        try: await interaction.followup.send("关闭频道失败：机器人缺少“管理频道”权限。", ephemeral=True)
        except discord.NotFound: pass
    except discord.NotFound:
        logger.warning(f"Channel {channel.id} not found during deletion, likely already deleted.")
    except discord.HTTPException as e:
        logger.error(f"HTTP error while deleting channel {channel.id}: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error during /close_chat command for channel {channel.id}")

# --- on_message 事件处理 (核心对话逻辑) ---
@bot.event
async def on_message(message: discord.Message):
    """处理在私密频道中的消息"""
    # 忽略自己、其他机器人或 Webhook 消息
    if message.author == bot.user or message.author.bot:
        return
    # 只处理特定前缀的文本频道
    if not isinstance(message.channel, discord.TextChannel) or not message.channel.name.startswith(PRIVATE_CHANNEL_PREFIX):
        return
    # 忽略空消息或可能的命令调用（以防万一）
    user_prompt = message.content.strip()
    if not user_prompt or user_prompt.startswith('/'): # 忽略空消息和看起来像命令的消息
        return

    channel_id = message.channel.id
    logger.info(f"Handling message in private channel {message.channel.name} ({channel_id}) from {message.author}: '{user_prompt[:50]}...'")

    # 获取或创建对话历史记录
    if channel_id not in conversation_history:
        conversation_history[channel_id] = deque(maxlen=MAX_HISTORY)
        logger.info(f"Initialized new history for channel {channel_id}")

    history_deque = conversation_history[channel_id]

    # 准备发送给 API 的消息列表 (只包含 role 和 content)
    api_messages = [{"role": msg.get("role"), "content": msg.get("content")} for msg in history_deque]
    api_messages.append({"role": "user", "content": user_prompt})

    # 调用 API 并处理响应
    try:
      async with message.channel.typing(): # 显示“正在输入...”
          # 使用 aiohttp session
          async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
              response_for_discord, response_for_history = await get_deepseek_response(session, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, api_messages)

      # 处理成功获取的响应
      if response_for_discord:
          # 只有包含有效最终回答时才更新历史
          if response_for_history:
                history_deque.append({"role": "user", "content": user_prompt})
                history_deque.append({"role": "assistant", "content": response_for_history})
                logger.debug(f"Added user & assistant turn to history for channel {channel_id}. History size: {len(history_deque)}")
          else:
                # 如果只有思维链，不添加到历史记录中，避免污染上下文
                logger.warning(f"Response for channel {channel_id} lacked final content. Not adding assistant turn to history.")

          # 发送消息并处理长度分割
          if len(response_for_discord) <= 2000:
              await message.channel.send(response_for_discord)
          else:
              logger.warning(f"Response for channel {channel_id} is too long ({len(response_for_discord)} chars), splitting.")
              parts = []
              current_pos = 0
              while current_pos < len(response_for_discord):
                    # 智能分割逻辑 (优先换行，其次空格，最后硬切，尝试保护代码块)
                    cut_off = min(current_pos + 1990, len(response_for_discord))
                    split_index = response_for_discord.rfind('\n', current_pos, cut_off)
                    if split_index == -1 or split_index <= current_pos:
                        space_split_index = response_for_discord.rfind(' ', current_pos, cut_off)
                        if space_split_index != -1 and space_split_index > current_pos: split_index = space_split_index
                        else: split_index = cut_off
                    chunk_to_check = response_for_discord[current_pos:split_index]
                    if "```" in chunk_to_check and chunk_to_check.count("```") % 2 != 0:
                        fallback_split = response_for_discord.rfind('\n', current_pos, split_index - 1)
                        if fallback_split != -1 and fallback_split > current_pos: split_index = fallback_split
                    parts.append(response_for_discord[current_pos:split_index])
                    current_pos = split_index
                    # 跳过分割点后的空白符
                    while current_pos < len(response_for_discord) and response_for_discord[current_pos].isspace(): current_pos += 1
              # 发送分割后的消息
              for i, part in enumerate(parts):
                  if not part.strip(): continue
                  if i > 0 and SPLIT_MESSAGE_DELAY > 0: await asyncio.sleep(SPLIT_MESSAGE_DELAY)
                  await message.channel.send(part.strip())

      elif response_for_history: # 如果 API 调用返回了错误信息
            await message.channel.send(f"抱歉，处理你的请求时发生错误：\n{response_for_history}")
      else: # 如果 API 调用既没返回成功也没返回错误信息（理论上不应发生）
          logger.error(f"Received unexpected None values from API call for channel {channel_id}.")
          await message.channel.send("抱歉，与 DeepSeek API 通信时发生未知问题。")

    # 处理发送消息时的异常
    except discord.Forbidden:
         logger.warning(f"Missing permissions (Forbidden) to send message in channel {channel_id}. Maybe channel deleted or permissions changed.")
    except discord.HTTPException as e:
         logger.error(f"Failed to send message to channel {channel_id} due to HTTPException: {e}")
    except Exception as e: # 捕获其他未预料的错误
        logger.exception(f"An unexpected error occurred in on_message handler for channel {channel_id}")
        try:
            # 尝试在频道内发送错误提示
            await message.channel.send(f"处理你的消息时发生内部错误，请稍后再试或联系管理员。错误：{e}")
        except Exception:
            # 如果连错误消息都发不出去，记录日志即可
            logger.error(f"Could not send the internal error message to channel {channel_id}.")


# --- 运行 Bot ---
if __name__ == "__main__":
    # 检查环境变量
    if not DISCORD_BOT_TOKEN:
        logger.critical("错误：未设置 DISCORD_BOT_TOKEN 环境变量！")
        exit("请设置 DISCORD_BOT_TOKEN 环境变量")
    if not DEEPSEEK_API_KEY:
        logger.critical("错误：未设置 DEEPSEEK_API_KEY 环境变量！")
        exit("请设置 DEEPSEEK_API_KEY 环境变量")

    try:
        logger.info("尝试启动 Discord 机器人 (commands.Bot)...")
        # 运行机器人, 禁用 discord.py 的默认日志处理器，避免重复日志
        bot.run(DISCORD_BOT_TOKEN, log_handler=None)
    except discord.LoginFailure:
        logger.critical("Discord Bot Token 无效，登录失败。请检查 Token。")
    except discord.PrivilegedIntentsRequired as e:
        logger.critical(f"必需的 Intents 未启用！请在 Discord开发者门户 开启 Message Content Intent。错误详情: {e}")
    except Exception as e:
        logger.exception("启动机器人时发生未捕获的错误。")