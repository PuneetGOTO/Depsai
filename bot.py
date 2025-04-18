import discord
from discord.ext import commands
from discord import app_commands
from discord.ui import View, Button, button
import os
import aiohttp
import json
import logging
from collections import deque
import asyncio
import re

# --- 日志记录设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- 配置 ---
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
# --- 确认模型支持 Vision，deepseek-chat 通常支持 ---
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat") 
logger.info(f"Initializing with DeepSeek Model: {DEEPSEEK_MODEL}")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
SPLIT_MESSAGE_DELAY = float(os.getenv("SPLIT_MESSAGE_DELAY", "0.3"))
admin_ids_str = os.getenv("ADMIN_ROLE_IDS", "")
ADMIN_ROLE_IDS = [int(role_id) for role_id in admin_ids_str.split(",") if role_id.strip().isdigit()]
PRIVATE_CHANNEL_PREFIX = "deepseek-"
# --- 新增：允许处理的最大图片附件数量 ---
MAX_IMAGE_ATTACHMENTS = int(os.getenv("MAX_IMAGE_ATTACHMENTS", 3)) # 限制每次处理的图片数量

# --- DeepSeek API 请求函数 (保持不变，调用方负责构建正确的 messages 格式) ---
async def get_deepseek_response(session, api_key, model, messages):
    """异步调用 DeepSeek API，现在可以处理包含图像 URL 的消息"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        # 根据 DeepSeek Vision 文档，可能需要调整 max_tokens 等参数
        # "max_tokens": 4096 # 视觉任务可能需要更多 token 输出
    }
    logger.info(f"Sending request to DeepSeek API using model '{model}' with {len(messages)} messages.")
    # 为了调试多模态输入，打印 payload 可能很有用
    # logger.debug(f"Request payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")

    try:
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=300) as response: # 增加超时
            raw_response_text = await response.text()
            try:
                response_data = json.loads(raw_response_text)
            except json.JSONDecodeError:
                 logger.error(f"Failed to decode JSON response. Status: {response.status}. Raw text: {raw_response_text[:500]}...")
                 return None, f"抱歉，无法解析 DeepSeek API 的响应 (状态码 {response.status})。"

            # logger.debug(f"Parsed response data: {json.dumps(response_data, indent=2, ensure_ascii=False)}")

            if response.status == 200:
                 if response_data.get("choices") and len(response_data["choices"]) > 0:
                    message_data = response_data["choices"][0].get("message", {})
                    # --- 注意：Vision 模型的响应可能没有 reasoning_content ---
                    # --- 我们直接取 content 作为主要回复 ---
                    final_content = message_data.get("content")
                    usage = response_data.get("usage")

                    # 检查是否有有效的回复内容
                    if not final_content:
                        logger.error("DeepSeek API response missing content.")
                        return None, "抱歉，DeepSeek API 返回的数据似乎不完整。"

                    logger.info(f"Successfully received response from DeepSeek. Usage: {usage}")
                    # 对于 Vision，我们主要关心最终的文本回复
                    # 返回两个值：用于显示的文本，和用于历史的相同文本
                    return final_content.strip(), final_content.strip()
                 else:
                     logger.error(f"DeepSeek API response missing 'choices': {response_data}")
                     return None, f"抱歉，DeepSeek API 返回了意外的结构：{response_data}"
            else:
                # 处理错误响应 (与之前相同)
                error_message = response_data.get("error", {}).get("message", f"未知错误 (状态码 {response.status})")
                # 检查是否是 Vision 输入相关的特定错误 (需要查阅 DeepSeek 文档)
                # 例如：无效的图片 URL，图片格式不支持等
                logger.error(f"DeepSeek API error (Status {response.status}): {error_message}. Response: {raw_response_text[:500]}")
                if response.status == 400:
                    error_message += "\n(提示: 错误 400 通常因为请求格式错误或输入无效，例如无法访问的图片URL)"
                return None, f"抱歉，调用 DeepSeek API 时出错 (状态码 {response.status}): {error_message}"
    # 处理网络和超时等异常 (与之前相同)
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
bot = commands.Bot(command_prefix="!", intents=intents)

# 对话历史
conversation_history = {}

# --- 创建按钮视图 (保持不变) ---
class CreateChatView(View):
    # ... (保持不变) ...
    @button(label="创建私密聊天", style=discord.ButtonStyle.primary, emoji="💬", custom_id="create_deepseek_chat_button")
    async def create_chat_button_callback(self, interaction: discord.Interaction, button_obj: Button):
        # ... (保持不变) ...
        guild = interaction.guild
        user = interaction.user
        bot_member = guild.get_member(bot.user.id) if guild else None
        if not guild or not bot_member:
            await interaction.response.send_message("无法获取服务器或机器人信息。", ephemeral=True)
            return
        if not bot_member.guild_permissions.manage_channels:
            await interaction.response.send_message("机器人缺少“管理频道”权限。", ephemeral=True)
            return
        clean_user_name = re.sub(r'[^\w-]', '', user.display_name.replace(' ', '-')).lower()
        if not clean_user_name: clean_user_name = "user"
        channel_name = f"{PRIVATE_CHANNEL_PREFIX}{clean_user_name}-{user.id % 10000}"
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(read_messages=False),
            user: discord.PermissionOverwrite(read_messages=True, send_messages=True, view_channel=True),
            bot_member: discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, manage_channels=True)
        }
        for role_id in ADMIN_ROLE_IDS:
            role = guild.get_role(role_id)
            if role: overwrites[role] = discord.PermissionOverwrite(read_messages=True, send_messages=True, manage_messages=True, view_channel=True)
        try:
            await interaction.response.send_message(f"正在创建频道 **{channel_name}** ...", ephemeral=True)
            category_name = "DeepSeek Chats"; category = discord.utils.find(lambda c: c.name.lower() == category_name.lower(), guild.categories)
            target_category = category if category and isinstance(category, discord.CategoryChannel) and category.permissions_for(bot_member).manage_channels else None
            new_channel = await guild.create_text_channel(channel_name, overwrites=overwrites, category=target_category)
            logger.info(f"Button Click: Created channel {new_channel.name} for user {user.name}")
            await new_channel.send(f"你好 {user.mention}！\n欢迎来到 DeepSeek 私密聊天频道 (模型: **{DEEPSEEK_MODEL}**)。\n直接输入问题或上传图片并提问。\n历史最多保留 **{MAX_HISTORY // 2}** 轮。\n完成后可用 `/close_chat` 关闭。")
            await interaction.followup.send(f"频道已创建：{new_channel.mention}", ephemeral=True)
        except Exception as e: logger.exception(f"Button Click: Error creating channel for {user.id}"); await interaction.followup.send("创建频道时出错。", ephemeral=True)


# --- setup_hook (保持不变) ---
@bot.event
async def setup_hook():
    # ... (保持不变) ...
    logger.info("Running setup_hook...")
    # ... (打印配置) ...
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash commands globally: {[c.name for c in synced]}")
    except Exception as e: logger.exception("Failed to sync slash commands")

# --- on_ready (保持不变) ---
@bot.event
async def on_ready():
    # ... (保持不变) ...
    logger.info(f'机器人已登录为 {bot.user}')
    print("Bot is ready and functional.")

# --- 斜杠命令 (保持不变) ---
# /setup_panel, /clear_history, /help, /close_chat
# ... (这些命令的代码保持不变，此处省略以减少篇幅) ...
@bot.tree.command(name="setup_panel", description="发送一个包含'创建聊天'按钮的消息到当前频道")
async def setup_panel(interaction: discord.Interaction, message_content: str = "点击下面的按钮开始与 DeepSeek 的私密聊天："):
    # ... (代码不变) ...
    channel = interaction.channel; user = interaction.user
    if not interaction.guild: await interaction.response.send_message("此命令只能在服务器频道中使用。", ephemeral=True); return
    is_private = isinstance(channel, discord.TextChannel) and channel.name.startswith(PRIVATE_CHANNEL_PREFIX)
    can_execute = is_private or (isinstance(user, discord.Member) and user.guild_permissions.manage_guild)
    if not can_execute: await interaction.response.send_message("你需要在非私密频道中拥有“管理服务器”权限。", ephemeral=True); return
    try:
        await channel.send(message_content, view=CreateChatView()); await interaction.response.send_message("按钮面板已发送！", ephemeral=True)
    except Exception as e: logger.exception(f"Failed setup panel in {channel.id}"); await interaction.response.send_message(f"发送面板出错: {e}", ephemeral=True)

@bot.tree.command(name="clear_history", description="清除当前私密聊天频道的对话历史")
async def clear_history(interaction: discord.Interaction):
    # ... (代码不变) ...
    channel = interaction.channel; user = interaction.user
    if not isinstance(channel, discord.TextChannel) or not channel.name.startswith(PRIVATE_CHANNEL_PREFIX): await interaction.response.send_message("此命令只能在私密频道中使用。", ephemeral=True); return
    channel_id = channel.id
    if channel_id in conversation_history:
        try: conversation_history[channel_id].clear(); logger.info(f"User {user} cleared history for {channel_id}"); await interaction.response.send_message("对话历史已清除。", ephemeral=False)
        except Exception as e: logger.exception(f"Error clearing history {channel_id}"); await interaction.response.send_message(f"清除历史出错: {e}", ephemeral=True)
    else: await interaction.response.send_message("未找到历史记录。", ephemeral=True)

@bot.tree.command(name="help", description="显示机器人使用帮助")
async def help_command(interaction: discord.Interaction):
    # ... (代码不变) ...
    embed = discord.Embed(title="DeepSeek 机器人帮助", description=f"模型: {DEEPSEEK_MODEL}", color=discord.Color.purple())
    embed.add_field(name="开始聊天", value="点击 **“创建私密聊天”** 按钮创建专属频道。", inline=False)
    embed.add_field(name="在私密频道中", value=f"• 直接输入问题或上传图片并提问。\n• 最多保留 **{MAX_HISTORY // 2}** 轮历史。", inline=False)
    embed.add_field(name="可用命令", value="`/help`: 显示帮助。\n`/clear_history`: 清除当前历史。\n`/close_chat`: 关闭当前频道。\n`/setup_panel`: 发送创建按钮面板。", inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.tree.command(name="close_chat", description="关闭当前的 DeepSeek 私密聊天频道")
async def close_chat(interaction: discord.Interaction):
    # ... (使用修正后的代码) ...
    channel = interaction.channel; user = interaction.user
    if not isinstance(channel, discord.TextChannel) or not channel.name.startswith(PRIVATE_CHANNEL_PREFIX): await interaction.response.send_message("此命令只能在私密频道中使用。", ephemeral=True); return
    try:
        await interaction.response.send_message(f"频道将在几秒后关闭...", ephemeral=True)
        await channel.send(f"此频道由 {user.mention} 请求关闭，将在 5 秒后删除。")
        channel_id = channel.id # 获取 ID
        if channel_id in conversation_history:
            try: del conversation_history[channel_id]; logger.info(f"Removed history for {channel_id}")
            except KeyError: logger.warning(f"History key {channel_id} already gone.")
        else: logger.warning(f"No history found for {channel_id} during closure.")
        await asyncio.sleep(5); await channel.delete(reason=f"Closed by {user}")
        logger.info(f"Deleted channel {channel.name} ({channel.id})")
    except Exception as e: logger.exception(f"Error closing channel {channel.id}")


# --- 修改后的 on_message 事件处理 ---
@bot.event
async def on_message(message: discord.Message):
    """处理在私密频道中接收到的消息，增加图片处理"""
    if message.author == bot.user or message.author.bot: return
    if not isinstance(message.channel, discord.TextChannel) or not message.channel.name.startswith(PRIVATE_CHANNEL_PREFIX): return
    # 忽略看起来像命令的消息
    if message.content.strip().startswith('/'): return

    channel_id = message.channel.id
    user_prompt_text = message.content.strip() # 获取文本内容

    # --- 新增：处理图片附件 ---
    image_urls = []
    if message.attachments:
        processed_images = 0
        for attachment in message.attachments:
            # 检查附件类型是否为图片，并限制处理数量
            if attachment.content_type and attachment.content_type.startswith("image/") and processed_images < MAX_IMAGE_ATTACHMENTS:
                image_urls.append(attachment.url)
                processed_images += 1
            elif processed_images >= MAX_IMAGE_ATTACHMENTS:
                 logger.warning(f"Reached max image limit ({MAX_IMAGE_ATTACHMENTS}) for message in channel {channel_id}. Ignoring further images.")
                 # 可以选择性地通知用户图片过多
                 # await message.channel.send(f"注意：本次只处理了前 {MAX_IMAGE_ATTACHMENTS} 张图片。")
                 break # 不再处理更多附件

    # --- 检查输入有效性 ---
    # 如果既没有文本也没有有效图片URL，则忽略
    if not user_prompt_text and not image_urls:
        # 如果只有无法识别的附件，可以选择忽略或提示
        # logger.debug(f"Ignoring message in {channel_id} with no text or processable images.")
        return

    # 如果只有图片没有文本，可以设置一个默认提示，或者要求用户必须输入文本
    if not user_prompt_text and image_urls:
        user_prompt_text = "描述这张/这些图片。" # 或者可以回复让用户提供问题
        # await message.reply("请提供关于图片的问题或描述要求。")
        # return

    logger.info(f"Handling message in {channel_id} from {message.author} with text: '{user_prompt_text[:50]}...' and {len(image_urls)} image(s).")

    # 获取或初始化历史
    if channel_id not in conversation_history:
        conversation_history[channel_id] = deque(maxlen=MAX_HISTORY)
    history_deque = conversation_history[channel_id]

    # --- 构建发送给 API 的消息列表 ---
    api_messages = []
    # 添加历史消息 (只添加文本部分，不包含历史图片)
    for msg in history_deque:
        # 确保只发送 role 和 content (content 应为字符串)
        # 如果历史记录格式复杂，需要在这里处理
        api_messages.append({"role": msg.get("role"), "content": msg.get("content")})

    # --- 构建当前用户的多模态消息 ---
    current_user_message_content = []
    # 添加文本部分
    current_user_message_content.append({
        "type": "text",
        "text": user_prompt_text
    })
    # 添加图片 URL 部分
    for url in image_urls:
        current_user_message_content.append({
            "type": "image_url",
            "image_url": {
                "url": url
                # 根据 DeepSeek 文档，可能可以指定 "detail": "low"/"high"
            }
        })

    # 将当前用户消息添加到 API 消息列表
    api_messages.append({"role": "user", "content": current_user_message_content})

    # --- 调用 API 并处理响应 (与之前类似，但历史记录处理有变化) ---
    try:
      async with message.channel.typing():
          async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
              # 注意：get_deepseek_response 现在返回 (显示的回复文本, 用于历史的回复文本)
              response_for_discord, response_for_history = await get_deepseek_response(session, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, api_messages)

      if response_for_discord:
          # --- 更新历史记录 (重要：简化版，只存文本) ---
          # 只保存用户输入的文本和模型回复的文本到历史记录
          # 不保存图片URL或多模态结构，避免历史记录过大和API请求复杂化
          history_deque.append({"role": "user", "content": user_prompt_text}) # 只存文本提示
          if response_for_history: # 确保有回复内容
                history_deque.append({"role": "assistant", "content": response_for_history}) # 存回复文本
                logger.debug(f"Added text-only user & assistant turn to history for {channel_id}. History size: {len(history_deque)}")
          else:
                # 如果 API 调用成功但没有返回文本 (不太可能，但做保护)
                 logger.warning(f"API call successful but no history content returned for {channel_id}.")


          # 发送并处理长消息分割 (与之前相同)
          if len(response_for_discord) <= 2000:
              await message.channel.send(response_for_discord)
          else:
              logger.warning(f"Response for channel {channel_id} is too long ({len(response_for_discord)} chars), splitting.")
              parts = []
              current_pos = 0
              # ... (省略详细分割代码, 与之前相同) ...
              while current_pos < len(response_for_discord):
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
                    while current_pos < len(response_for_discord) and response_for_discord[current_pos].isspace(): current_pos += 1
              for i, part in enumerate(parts):
                  if not part.strip(): continue
                  if i > 0 and SPLIT_MESSAGE_DELAY > 0: await asyncio.sleep(SPLIT_MESSAGE_DELAY)
                  await message.channel.send(part.strip())

      elif response_for_history: # API 返回错误信息
            await message.channel.send(f"抱歉，处理请求时出错：\n{response_for_history}")
      else: # 未知错误
          logger.error(f"Received unexpected None values from API call for channel {channel_id}.")
          await message.channel.send("抱歉，与 DeepSeek API 通信时未知问题。")
    # 处理发送消息时的异常 (与之前相同)
    except discord.Forbidden: logger.warning(f"Missing permissions in channel {channel_id}.")
    except discord.HTTPException as e: logger.error(f"HTTPException sending message to channel {channel_id}: {e}")
    except Exception as e:
        logger.exception(f"Error in on_message handler for channel {channel_id}")
        try: await message.channel.send(f"处理消息时内部错误: {e}")
        except Exception: pass

# --- 运行 Bot ---
if __name__ == "__main__":
    # ... (启动检查保持不变) ...
    if not DISCORD_BOT_TOKEN: logger.critical("未设置 DISCORD_BOT_TOKEN"); exit(1)
    if not DEEPSEEK_API_KEY: logger.critical("未设置 DEEPSEEK_API_KEY"); exit(1)
    try:
        logger.info("尝试启动 Discord 机器人 (commands.Bot)...")
        bot.run(DISCORD_BOT_TOKEN, log_handler=None)
    except discord.LoginFailure: logger.critical("Token 无效，登录失败。")
    except discord.PrivilegedIntentsRequired as e: logger.critical(f"必需 Intents 未启用！请开启 Message Content Intent。错误: {e}")
    except Exception as e: logger.exception("启动机器人时发生错误。")